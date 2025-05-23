# GenDeblur
Library for researching of genetic algorithms application for blind image deconvolution.
Currently works only with artificially (programmatically) blurred images.

Example of usage: https://github.com/EpigramxD/GendeblurExample

PyPi package: https://pypi.org/project/gendeblur/

Examples:

sharp image, psf, image blurred with psf:
![aHaGbdfU8U_rdTRkFCjOEOCpNSQwWOGzrbLjdG7BPdWjIBIBQHb07aKIYzqvAeokBjNv2bWfl-Va1CQz1yc3N6Nd](https://user-images.githubusercontent.com/52430062/170833691-86d80321-ce0a-445f-af1d-4332c466978d.jpg)
estimated psf and deblurred image:
![dxY7_HdZIXmY1AjWgPdlwY7fB8Oc9Qs0c554Se2PXnuSOFKXGbOv042x6OKNDnBZI59nAKRa8VAbdKtwH0RAyhUo](https://user-images.githubusercontent.com/52430062/170833695-09d61e6b-25c9-4da4-9da8-d84950e47883.jpg)
sharp image, psf, image blurred with psf:
![lTVUBLZesSsTLSBnKKWmYJ0AdYOZyK1iUo1dhxOCIbILlog3KHC2V3RI2GBCuSgzo3vDs5Y49RPTBW92kcHsIgoz](https://user-images.githubusercontent.com/52430062/170833702-aaa4a348-c77b-4497-87c9-e7291bba0fe2.jpg)
estimated psf and deblurred image:
![kDc1msqSmz_k9aWxnTf-2cVSnmHnbwhapK0DnCagDhxxFjhyN1LoOVRiwyCDSxRh9Jk30Eqkrvp4OWas1K7i699J](https://user-images.githubusercontent.com/52430062/170833704-1d16ff27-453a-4050-bec6-21fbb3841b75.jpg)


## Requirements:
- Python 3.8.0
- All packages from requirements.txt

## Installation:
```
pip install -r requirements.txt
pip install gendeblur
```
or
```
pip install git+https://github.com/EpigramxD/GenDeblur.git
```

Build package: ```python -m build```

Push package to PyPi: ```python -m twine upload --verbose --repository pypi dist/*```

