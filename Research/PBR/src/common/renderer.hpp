/*
 * Physically Based Rendering
 * Copyright (c) 2017-2018 Michał Siejak
 */

#pragma once
#include <glm/mat4x4.hpp>

struct GLFWwindow;

// Material package directory (trainer output: classic BC1 maps + neural
// latents/weights in one folder), relative to the data/ working directory or
// absolute. Set from the -material command line argument.
#include <string>
extern std::string g_materialDir;

struct ViewSettings
{
	float pitch = 0.0f;
	float yaw = 0.0f;
	float distance;
	float fov;
};

struct SceneSettings
{
	float pitch = 0.0f;
	float yaw = 0.0f;

	static const int NumLights = 3;
	struct Light {
		glm::vec3 direction;
		glm::vec3 radiance;
		bool enabled = false;
	} lights[NumLights];

	// 0=classic textures, 1=neural material, 2=B/W difference (classic vs neural).
	// N toggles 0<->1, D toggles the difference view. Forced to 0 when no
	// neural assets are loaded.
	int materialMode = 1;

	// MLP backend for the neural material: false = plain FMA (SM 5.0),
	// true = D3D12 Linear Algebra / cooperative vectors (SM 6.10 preview).
	// C toggles; ignored when the LinAlg backend is unavailable.
	bool useLinAlg = false;
};

class RendererInterface
{
public:
	virtual ~RendererInterface() = default;

	virtual GLFWwindow* initialize(int width, int height, int maxSamples) = 0;
	virtual void shutdown() = 0;
	virtual void setup() = 0;
	virtual void render(GLFWwindow* window, const ViewSettings& view, const SceneSettings& scene) = 0;
};
