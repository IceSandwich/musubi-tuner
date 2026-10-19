# Requirements
- triton (for compile only, pls install `triton-windows` on Windows)
- comfy environment (via venv)

# Training
## Attention
Pick one of them:
1. Use sdpa only
2. Use flashattention + compile (since compile doesn't support sdpa for now)

# Troubleshooting
# Compile encode issue
```
Traceback (most recent call last):
  File "E:\Projects\Github\musubi-tuner\krea2_train_network.py", line 12, in <module>
    main()
  File "E:\Projects\Github\musubi-tuner\src\musubi_tuner\krea2_train_network.py", line 614, in main
    trainer.train(args)
  File "E:\Projects\Github\musubi-tuner\src\musubi_tuner\training\trainer_base.py", line 1494, in train
    ) = self._prepare_with_accelerator(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Projects\Github\musubi-tuner\src\musubi_tuner\training\trainer_base.py", line 1842, in _prepare_with_accelerator
    transformer = self.compile_transformer(args, transformer)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Projects\Github\musubi-tuner\src\musubi_tuner\krea2_train_network.py", line 462, in compile_transformer
    return model_utils.compile_transformer(args, model, [model.blocks], disable_linear=disable_linear)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Projects\Github\musubi-tuner\src\musubi_tuner\utils\model_utils.py", line 334, in compile_transformer
    block = torch.compile(
            ^^^^^^^^^^^^^^
  File "D:\Projects\miniconda3\envs\py312torch213cu132\Lib\site-packages\torch\__init__.py", line 2836, in compile
    return torch._dynamo.optimize(
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Projects\miniconda3\envs\py312torch213cu132\Lib\site-packages\torch\_dynamo\eval_frame.py", line 1682, in optimize
    return _optimize(rebuild_ctx, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Projects\miniconda3\envs\py312torch213cu132\Lib\site-packages\torch\_dynamo\eval_frame.py", line 1799, in _optimize
    backend.get_compiler_config()
  File "D:\Projects\miniconda3\envs\py312torch213cu132\Lib\site-packages\torch\__init__.py", line 2497, in get_compiler_config
    from torch._inductor.compile_fx import get_patched_config_dict
  File "D:\Projects\miniconda3\envs\py312torch213cu132\Lib\site-packages\torch\_inductor\compile_fx.py", line 114, in <module>
    from . import config, distributed_autotune, metrics
  File "D:\Projects\miniconda3\envs\py312torch213cu132\Lib\site-packages\torch\_inductor\distributed_autotune.py", line 15, in <module>
    from . import config, select_algorithm
  File "D:\Projects\miniconda3\envs\py312torch213cu132\Lib\site-packages\torch\_inductor\select_algorithm.py", line 6073, in <module>
    from . import lowering  # noqa: F401
    ^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Projects\miniconda3\envs\py312torch213cu132\Lib\site-packages\torch\_inductor\lowering.py", line 9111, in <module>
    import_submodule(kernel)
  File "D:\Projects\miniconda3\envs\py312torch213cu132\Lib\site-packages\torch\_dynamo\utils.py", line 4384, in import_submodule
    importlib.import_module(f"{mod.__name__}.{filename[:-3]}")
  File "D:\Projects\miniconda3\envs\py312torch213cu132\Lib\importlib\__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Projects\miniconda3\envs\py312torch213cu132\Lib\site-packages\torch\_inductor\kernel\mm_grouped.py", line 142, in <module>
    source=load_kernel_template("cutedsl_mm_grouped"),
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Projects\miniconda3\envs\py312torch213cu132\Lib\site-packages\torch\_inductor\utils.py", line 4770, in load_template
    return f.read()
           ^^^^^^^^
UnicodeDecodeError: 'gbk' codec can't decode byte 0x94 in position 618: illegal multibyte sequence
```
Because system use gbk in Windows by default, please set environment variable:
```
set PYTHONUTF8=1
```
