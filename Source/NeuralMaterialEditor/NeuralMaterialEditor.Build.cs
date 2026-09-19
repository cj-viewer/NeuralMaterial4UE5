using UnrealBuildTool;

public class NeuralMaterialEditor : ModuleRules
{
    public NeuralMaterialEditor(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PrivateDependencyModuleNames.AddRange(new[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "UnrealEd",
            "NeuralMaterialRuntime"
        });
    }
}
