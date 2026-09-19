#include "Modules/ModuleManager.h"

class FNeuralMaterialEditorModule final : public IModuleInterface
{
public:
    virtual void StartupModule() override {}
    virtual void ShutdownModule() override {}
};

IMPLEMENT_MODULE(FNeuralMaterialEditorModule, NeuralMaterialEditor)
