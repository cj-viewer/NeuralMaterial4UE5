// Physically Based Rendering
// Copyright (c) 2017-2018 Michał Siejak

// Physically Based shading model: Lambetrtian diffuse BRDF + Cook-Torrance microfacet specular BRDF + IBL for ambient.

// This implementation is based on "Real Shading in Unreal Engine 4" SIGGRAPH 2013 course notes by Epic Games.
// See: http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf

static const float PI = 3.141592;
static const float Epsilon = 0.00001;

static const uint NumLights = 3;

// Constant normal incidence Fresnel factor for all dielectrics.
static const float3 Fdielectric = 0.04;

cbuffer TransformConstants : register(b0)
{
	float4x4 viewProjectionMatrix;
	float4x4 skyProjectionMatrix;
	float4x4 sceneRotationMatrix;
};

cbuffer ShadingConstants : register(b0)
{
	struct {
		float3 direction;
		float3 radiance;
	} lights[NumLights];
	float3 eyePosition;
	float materialMode; // packed into eyePosition.w: 0=classic, 1=neural, 2=B/W difference
};

static const float DiffAmplify = 8.0;

// Neural material: 4 BC1 latent textures + 12->32->9 MLP (NeuralMaterial4UE5 M1/M2 export).
// Weight layout in the packed float stream: w0[32][12] | b0[32] | w1[9][32] | b1[9].
cbuffer NeuralWeights : register(b1)
{
	float4 neuralWeights[179]; // 716 floats (713 used + padding)
};

struct VertexShaderInput
{
	float3 position  : POSITION;
	float3 normal    : NORMAL;
	float3 tangent   : TANGENT;
	float3 bitangent : BITANGENT;
	float2 texcoord  : TEXCOORD;
};
struct PixelShaderInput
{
	float4 pixelPosition : SV_POSITION;
	float3 position : POSITION;
	float2 texcoord : TEXCOORD;
	float3x3 tangentBasis : TBASIS;
};

Texture2D albedoTexture : register(t0);
Texture2D normalTexture : register(t1);
Texture2D metalnessTexture : register(t2);
Texture2D roughnessTexture : register(t3);
TextureCube specularTexture : register(t4);
TextureCube irradianceTexture : register(t5);
Texture2D specularBRDF_LUT : register(t6);

Texture2D latentTexture0 : register(t7);
Texture2D latentTexture1 : register(t8);
Texture2D latentTexture2 : register(t9);
Texture2D latentTexture3 : register(t10);

#if USE_LINALG
// SM 6.9 cooperative-vector MLP backend (D3D12 preview), compiled with DXC.
// fp16 weight buffer layout (bytes): W0 32x12 row-major stride 24 @0,
// b0 @1024, W1 9x32 row-major stride 64 @1152, b1 @1792.
#include <dx/linalg.h>
using namespace dx::linalg;
ByteAddressBuffer neuralWeightsFP16 : register(t11);
#endif

SamplerState defaultSampler : register(s0);
SamplerState spBRDF_Sampler : register(s1);

// GGX/Towbridge-Reitz normal distribution function.
// Uses Disney's reparametrization of alpha = roughness^2.
float ndfGGX(float cosLh, float roughness)
{
	float alpha   = roughness * roughness;
	float alphaSq = alpha * alpha;

	float denom = (cosLh * cosLh) * (alphaSq - 1.0) + 1.0;
	return alphaSq / (PI * denom * denom);
}

// Single term for separable Schlick-GGX below.
float gaSchlickG1(float cosTheta, float k)
{
	return cosTheta / (cosTheta * (1.0 - k) + k);
}

// Schlick-GGX approximation of geometric attenuation function using Smith's method.
float gaSchlickGGX(float cosLi, float cosLo, float roughness)
{
	float r = roughness + 1.0;
	float k = (r * r) / 8.0; // Epic suggests using this roughness remapping for analytic lights.
	return gaSchlickG1(cosLi, k) * gaSchlickG1(cosLo, k);
}

// Shlick's approximation of the Fresnel factor.
float3 fresnelSchlick(float3 F0, float cosTheta)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float neuralWeight(uint i)
{
	return neuralWeights[i >> 2][i & 3];
}

float srgbToLinear(float c)
{
	return (c <= 0.04045) ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

// Evaluate the neural material: sample the 4 BC1 latent mip chains with the
// ordinary hardware sampler (decode + trilinear/aniso happen in the texture
// unit, exactly as trained), then run the 12->32->9 FMA MLP.
// Outputs (storage space): 0-2 basecolor sRGB, 3-5 normal, 6 AO, 7 roughness, 8 metallic.
void neuralMaterial(float2 uv, out float3 albedo, out float3 tangentNormal,
                    out float ao, out float roughness, out float metalness)
{
	float x[12];
	float3 f0 = latentTexture0.Sample(defaultSampler, uv).rgb;
	float3 f1 = latentTexture1.Sample(defaultSampler, uv).rgb;
	float3 f2 = latentTexture2.Sample(defaultSampler, uv).rgb;
	float3 f3 = latentTexture3.Sample(defaultSampler, uv).rgb;
	x[0] = f0.r; x[1]  = f0.g; x[2]  = f0.b;
	x[3] = f1.r; x[4]  = f1.g; x[5]  = f1.b;
	x[6] = f2.r; x[7]  = f2.g; x[8]  = f2.b;
	x[9] = f3.r; x[10] = f3.g; x[11] = f3.b;

#if USE_LINALG
	vector<half, 12> xh;
	[unroll]
	for(uint i = 0; i < 12; ++i) {
		xh[i] = (half)x[i];
	}

	MatrixRef<DATA_TYPE_FLOAT16, 32, 12, MATRIX_LAYOUT_ROW_MAJOR> w0m = {neuralWeightsFP16, 0, 24};
	VectorRef<DATA_TYPE_FLOAT16> b0v = {neuralWeightsFP16, 1024};
	vector<half, 32> hiddenV =
		MulAdd<half>(w0m, MakeInterpretedVector<DATA_TYPE_FLOAT16>(xh), b0v);
	hiddenV = max(hiddenV, (half)0.0);

	MatrixRef<DATA_TYPE_FLOAT16, 9, 32, MATRIX_LAYOUT_ROW_MAJOR> w1m = {neuralWeightsFP16, 1152, 64};
	VectorRef<DATA_TYPE_FLOAT16> b1v = {neuralWeightsFP16, 1792};
	vector<half, 9> yh =
		MulAdd<half>(w1m, MakeInterpretedVector<DATA_TYPE_FLOAT16>(hiddenV), b1v);

	float y[9];
	[unroll]
	for(uint o = 0; o < 9; ++o) {
		y[o] = (float)yh[o];
	}
#else
	float hidden[32];
	[unroll]
	for(uint j = 0; j < 32; ++j) {
		float v = neuralWeight(384 + j); // b0
		[unroll]
		for(uint i = 0; i < 12; ++i) {
			v = mad(x[i], neuralWeight(j * 12 + i), v); // w0
		}
		hidden[j] = max(v, 0.0);
	}

	float y[9];
	[unroll]
	for(uint o = 0; o < 9; ++o) {
		float v = neuralWeight(704 + o); // b1
		[unroll]
		for(uint j = 0; j < 32; ++j) {
			v = mad(hidden[j], neuralWeight(416 + o * 32 + j), v); // w1
		}
		y[o] = v;
	}
#endif

	// The MLP reproduces storage-space texture values; convert exactly like the
	// classic path does (albedo texture is an sRGB view, normal is *2-1).
	albedo = float3(srgbToLinear(saturate(y[0])), srgbToLinear(saturate(y[1])), srgbToLinear(saturate(y[2])));
	tangentNormal = 2.0 * saturate(float3(y[3], y[4], y[5])) - 1.0;
	ao = saturate(y[6]);
	roughness = saturate(y[7]);
	metalness = saturate(y[8]);
}

// Returns number of mipmap levels for specular IBL environment map.
uint querySpecularTextureLevels()
{
	uint width, height, levels;
	specularTexture.GetDimensions(0, width, height, levels);
	return levels;
}

// Vertex shader
PixelShaderInput main_vs(VertexShaderInput vin)
{
	PixelShaderInput vout;
	vout.position = mul(sceneRotationMatrix, float4(vin.position, 1.0)).xyz;
	vout.texcoord = float2(vin.texcoord.x, 1.0-vin.texcoord.y);

	// Pass tangent space basis vectors (for normal mapping).
	float3x3 TBN = float3x3(vin.tangent, vin.bitangent, vin.normal);
	vout.tangentBasis = mul((float3x3)sceneRotationMatrix, transpose(TBN));

	float4x4 mvpMatrix = mul(viewProjectionMatrix, sceneRotationMatrix);
	vout.pixelPosition = mul(mvpMatrix, float4(vin.position, 1.0));
	return vout;
}

void classicMaterial(float2 uv, out float3 albedo, out float3 tangentNormal,
                     out float ao, out float roughness, out float metalness)
{
	albedo = albedoTexture.Sample(defaultSampler, uv).rgb;
	metalness = metalnessTexture.Sample(defaultSampler, uv).r;
	roughness = roughnessTexture.Sample(defaultSampler, uv).r;
	tangentNormal = 2.0 * normalTexture.Sample(defaultSampler, uv).rgb - 1.0;
	ao = 1.0; // the Cerberus set has no AO map
}

// Full shading (direct + IBL) for one set of material parameters.
float3 shade(PixelShaderInput pin, float3 albedo, float3 tangentNormal,
             float ao, float roughness, float metalness)
{
	// Outgoing light direction (vector from world-space fragment position to the "eye").
	float3 Lo = normalize(eyePosition - pin.position);

	// Get current fragment's normal and transform to world space.
	float3 N = normalize(tangentNormal);
	N = normalize(mul(pin.tangentBasis, N));
	
	// Angle between surface normal and outgoing light direction.
	float cosLo = max(0.0, dot(N, Lo));
		
	// Specular reflection vector.
	float3 Lr = 2.0 * cosLo * N - Lo;

	// Fresnel reflectance at normal incidence (for metals use albedo color).
	float3 F0 = lerp(Fdielectric, albedo, metalness);

	// Direct lighting calculation for analytical lights.
	float3 directLighting = 0.0;
	for(uint i=0; i<NumLights; ++i)
	{
		float3 Li = -lights[i].direction;
		float3 Lradiance = lights[i].radiance;

		// Half-vector between Li and Lo.
		float3 Lh = normalize(Li + Lo);

		// Calculate angles between surface normal and various light vectors.
		float cosLi = max(0.0, dot(N, Li));
		float cosLh = max(0.0, dot(N, Lh));

		// Calculate Fresnel term for direct lighting. 
		float3 F  = fresnelSchlick(F0, max(0.0, dot(Lh, Lo)));
		// Calculate normal distribution for specular BRDF.
		float D = ndfGGX(cosLh, roughness);
		// Calculate geometric attenuation for specular BRDF.
		float G = gaSchlickGGX(cosLi, cosLo, roughness);

		// Diffuse scattering happens due to light being refracted multiple times by a dielectric medium.
		// Metals on the other hand either reflect or absorb energy, so diffuse contribution is always zero.
		// To be energy conserving we must scale diffuse BRDF contribution based on Fresnel factor & metalness.
		float3 kd = lerp(float3(1, 1, 1) - F, float3(0, 0, 0), metalness);

		// Lambert diffuse BRDF.
		// We don't scale by 1/PI for lighting & material units to be more convenient.
		// See: https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
		float3 diffuseBRDF = kd * albedo;

		// Cook-Torrance specular microfacet BRDF.
		float3 specularBRDF = (F * D * G) / max(Epsilon, 4.0 * cosLi * cosLo);

		// Total contribution for this light.
		directLighting += (diffuseBRDF + specularBRDF) * Lradiance * cosLi;
	}

	// Ambient lighting (IBL).
	float3 ambientLighting;
	{
		// Sample diffuse irradiance at normal direction.
		float3 irradiance = irradianceTexture.Sample(defaultSampler, N).rgb;

		// Calculate Fresnel term for ambient lighting.
		// Since we use pre-filtered cubemap(s) and irradiance is coming from many directions
		// use cosLo instead of angle with light's half-vector (cosLh above).
		// See: https://seblagarde.wordpress.com/2011/08/17/hello-world/
		float3 F = fresnelSchlick(F0, cosLo);

		// Get diffuse contribution factor (as with direct lighting).
		float3 kd = lerp(1.0 - F, 0.0, metalness);

		// Irradiance map contains exitant radiance assuming Lambertian BRDF, no need to scale by 1/PI here either.
		float3 diffuseIBL = kd * albedo * irradiance;

		// Sample pre-filtered specular reflection environment at correct mipmap level.
		uint specularTextureLevels = querySpecularTextureLevels();
		float3 specularIrradiance = specularTexture.SampleLevel(defaultSampler, Lr, roughness * specularTextureLevels).rgb;

		// Split-sum approximation factors for Cook-Torrance specular BRDF.
		float2 specularBRDF = specularBRDF_LUT.Sample(spBRDF_Sampler, float2(cosLo, roughness)).rg;

		// Total specular IBL contribution.
		float3 specularIBL = (F0 * specularBRDF.x + specularBRDF.y) * specularIrradiance;

		// Total ambient lighting contribution (AO comes from the neural output;
		// the classic Cerberus set has no AO map, so ao == 1 on that path).
		ambientLighting = (diffuseIBL + specularIBL) * ao;
	}

	return directLighting + ambientLighting;
}

// Pixel shader
float4 main_ps(PixelShaderInput pin) : SV_Target
{
	float3 albedo, tangentNormal;
	float ao, roughness, metalness;

	if(materialMode > 1.5) {
		// B/W difference mode: shade both material paths and output the
		// amplified luminance of the absolute difference (pre-tonemap).
		classicMaterial(pin.texcoord, albedo, tangentNormal, ao, roughness, metalness);
		float3 classicColor = shade(pin, albedo, tangentNormal, ao, roughness, metalness);
		neuralMaterial(pin.texcoord, albedo, tangentNormal, ao, roughness, metalness);
		float3 neuralColor = shade(pin, albedo, tangentNormal, ao, roughness, metalness);
		float g = dot(abs(neuralColor - classicColor), float3(0.299, 0.587, 0.114)) * DiffAmplify;
		return float4(g, g, g, 1.0);
	}

	if(materialMode > 0.5) {
		neuralMaterial(pin.texcoord, albedo, tangentNormal, ao, roughness, metalness);
	}
	else {
		classicMaterial(pin.texcoord, albedo, tangentNormal, ao, roughness, metalness);
	}
	return float4(shade(pin, albedo, tangentNormal, ao, roughness, metalness), 1.0);
}
