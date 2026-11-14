# How to build the documentation

This documentation is built with [Documenteer](https://documenteer.lsst.io), which wraps [Sphinx](http://www.sphinx-doc.org/en/stable/). To install the documentation build dependencies:

```sh
pip install -r requirements.txt
```

Documenteer reads the project's name, version and source URL from the installed distribution metadata, so make that exist first, either by installing the package (`pip install -e ..`) or by running `scons`, whose `pkginfo` step generates both `version.py` and a `python/lsst_gkcoadd.dist-info` directory.

[Build the documentation with Documenteer](https://documenteer.lsst.io/pipelines/package-docs-cli.html):

```sh
package-docs build
```

If necessary, you can also clean the built documentation and intermediate products:

```sh
package-docs clean
```

The published copy of this documentation is built with the rest of the Science Pipelines documentation on [LSST the Docs](https://pipelines.lsst.io); the single-package build here is for previewing it.
