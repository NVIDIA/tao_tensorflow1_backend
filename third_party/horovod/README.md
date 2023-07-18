# Upgrading horovod

## Objective

Describe procedure for updating horovod, using horovod==0.22.1 upgrade as an example.

## Building horovod

Version 0.22.1 of Horovod has a bug in the keras allreduce function where it doesn't pass 2 keyword arguments, resulting in horovod.keras.all_reduce() failing. As a workaround to this, TAO toolkit patches horovod from source and builds a custom package of horovod.

### Build command

Use the build command below to build horovod wheel in the `tao-toolkit-tf` base docker environent.

```sh
dazel run third_party/horovod:build_horovod
```

Once this command finishes, it would output a link to the built wheel in the bazel cache, which you may upload to the `tlt-tf` pypi index by running something like

```sh
TWINE_USERNAME=__token__ TWINE_PASSWORD=<personal_access_token> twine upload --repository url <url>
```

For testing purposes, you may save the wheel locally at `</somewhere/on/disk>/horovod-0.22.1-cp36-cp36m-linux_x86_64.whl` and add the location of the build wheel in `tlt-tf/ci/runtime_resources/requirements-pip.txt`, add path to the directory where wheel is located to `WORKSPACE`, in the `pip_import` block,

```sh
        "--find-links=file://</somewhere/on/disk>"
```

Once this is done, we can run

```sh
dazel run @pip_deps//:update
```

to update the `requriements-pip.bzl` file and launch some validation workflows. After verification, the wheel can be uploaded to the `tlt-tf` gitlab pypi as pointed out earlier by the wheel build process and replace the local link with a link to remote location of the horovod wheel in `requirements-pip.txt`, and also remove the modification made in `WORKSPACE` file and run

```sh
dazen run @pip_deps//:update
```

again.
