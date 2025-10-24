# Add your finetuning pipeline here
# It should contains a function train() with params:
# pipeline.train(
#     model_name=model_name,
#     model_dir=model_dir,
#     dataset=dataset,
#     dataset_extras=dataset_extras,
#     dataset_format=kwargs.get("dataset_format"),
# )
# Make sure to add it to core/args.py as well
# Select your pipeline with --pipeline <name_of_the_pipeline_file>
