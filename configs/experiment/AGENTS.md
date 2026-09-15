# Experiment Configuration Contract

Every configuration in this directory must represent a scientifically identifiable experiment.

Before creating or editing an experiment configuration:

1. identify the parent or reference experiment;
2. state the scientific hypothesis;
3. state the exact configuration keys allowed to differ;
4. inspect the resolved configuration, not only the YAML text;
5. preserve every unrelated setting.

For a one-factor ablation:

`Delta(actual resolved config) = Delta(declared ablation)`

Never silently change the dataset or split, input resolution, preprocessing, seed policy, optimizer, learning-rate schedule, training duration, teacher pool, checkpoint-selection rule, evaluation datasets, threshold, or metric definitions unless that factor is explicitly under study.

Do not infer execution status from the existence of a configuration. A future registered experiment must have a unique experiment ID and must not overwrite a scientifically different experiment.

If the intended scientific change cannot be represented without changing additional factors, stop and report the dependency instead of hiding it.
