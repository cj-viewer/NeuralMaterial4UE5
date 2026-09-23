/*
 * Physically Based Rendering
 * Copyright (c) 2017-2018 Michał Siejak
 */

#if !(defined(ENABLE_OPENGL) || defined(ENABLE_VULKAN) || defined(ENABLE_D3D11) || defined(ENABLE_D3D12))
#error "At least one renderer implementation must be enabled via an appropriate ENABLE_* preprocessor macro"
#endif

#include <cstdio>
#include <cstdlib>
#include <string>
#include <memory>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "application.hpp"

#include "../opengl.hpp"
#include "../vulkan.hpp"
#include "../d3d11.hpp"
#include "../d3d12.hpp"

static void printUsage(const char* argv0)
{
	const std::vector<const char*> flags = {
#if defined(ENABLE_OPENGL)
		"-opengl",
#endif
#if defined(ENABLE_VULKAN)
		"-vulkan",
#endif
#if defined(ENABLE_D3D11)
		"-d3d11",
#endif
#if defined(ENABLE_D3D12)
		"-d3d12",
#endif
	};

	std::fprintf(stderr, "Usage: %s [", argv0);
	for(size_t i=0; i<flags.size(); ++i) {
		std::fprintf(stderr, "%s%s", flags[i], i < (flags.size()-1) ? "|":"");
	}
	std::fprintf(stderr, "] [-datadir <dir>] [-material <dir>]\n");
	std::fprintf(stderr, "  -datadir   Renderer asset root (the PBR/data folder). The process\n"
	                     "             chdirs there, so the exe can be launched from anywhere.\n");
	std::fprintf(stderr, "  -material  Material package folder (trainer output): classic BC1 maps\n"
	                     "             + neural latents/weights. Resolved against the launch\n"
	                     "             directory when given; default \"material\" inside datadir.\n");
}

static RendererInterface* createDefaultRenderer()
{
#if defined(ENABLE_D3D11)
	return new D3D11::Renderer;
#elif defined(ENABLE_D3D12)
	return new D3D12::Renderer;
#elif defined(ENABLE_OPENGL)
	return new OpenGL::Renderer;
#elif defined(ENABLE_VULKAN)
	return new Vulkan::Renderer;
#endif
}

static RendererInterface* createNamedRenderer(const std::string& flag)
{
#if defined(ENABLE_OPENGL)
	if(flag == "-opengl") {
		return new OpenGL::Renderer;
	}
#endif
#if defined(ENABLE_VULKAN)
	if(flag == "-vulkan") {
		return new Vulkan::Renderer;
	}
#endif
#if defined(ENABLE_D3D11)
	if(flag == "-d3d11") {
		return new D3D11::Renderer;
	}
#endif
#if defined(ENABLE_D3D12)
	if(flag == "-d3d12") {
		return new D3D12::Renderer;
	}
#endif
	return nullptr;
}

int main(int argc, char* argv[])
{
	RendererInterface* renderer = nullptr;
	std::string dataDir;
	bool materialSet = false;

	for(int i=1; i<argc; ++i) {
		const std::string arg = argv[i];
		if(arg == "-material" && i + 1 < argc) {
			g_materialDir = argv[++i];
			materialSet = true;
		}
		else if(arg == "-datadir" && i + 1 < argc) {
			dataDir = argv[++i];
		}
		else {
			renderer = createNamedRenderer(arg);
			if(!renderer) {
				printUsage(argv[0]);
				return 1;
			}
		}
	}
	if(!renderer) {
		renderer = createDefaultRenderer();
	}

#ifdef _WIN32
	// Resolve an explicit -material against the launch directory BEFORE
	// chdir'ing into the asset root, so both paths are independent.
	if(materialSet) {
		char full[MAX_PATH];
		if(_fullpath(full, g_materialDir.c_str(), MAX_PATH)) {
			g_materialDir = full;
		}
	}
	if(!dataDir.empty() && !SetCurrentDirectoryA(dataDir.c_str())) {
		std::fprintf(stderr, "Error: -datadir \"%s\" is not accessible\n", dataDir.c_str());
		return 1;
	}
#endif
	std::printf("Material package: %s\n", g_materialDir.c_str());

	try {
		Application().run(std::unique_ptr<RendererInterface>{ renderer });
	}
	catch(const std::exception& e) {
		std::fprintf(stderr, "Error: %s\n", e.what());
		return 1;
	}
}
