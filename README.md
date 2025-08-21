# pysarflow

An open-source python library for SAR data processing

> [!WARNING]
> This library is under active development and lot of its functionality is still yet to code.


## Description

Despite the growing usage of SAR data, the processing workflow remains complex and still relies on specialized heavy desktop tools creating huge barriers for students and non-experts. There is a growing need for a modular, open-source and user-friendly workflow that can take Level-1 SAR products and guide users through essential steps ideally ending in products that are usable and compatible with geospatial pipelines.

## Project Structure

```graphql
pysarflow/
├── __init__.py
├── grd.py
├── slc.py
├── utils.py

```

## Installation

To use this package, the esa-snappy library (the Python interface for ESA SNAP) is required before installation of the pysarflow package.  
The follwoing are the steps to install exsa-snappy :

**1. Create and activate a conda environment**   
```bash
conda create -n snap_env python=3.9   
conda activate snap_env  
```
**2. Install Package**
```bash
pip install pysarflow 
```

**3. Install ESA SNAP Desktop**    
Download and install ESA SNAP from the [SNAP website](https://earth.esa.int/eogateway/tools/snap).  
During installation, enable the option to configure Python for SNAP and specify your Python executable path:  
- Use the Python from your conda environment, e.g. *C:\Users\YourUsername\.conda\envs\snap_env\python.exe*  
- If that does not work, try the base environment Python: for example *(C:\ProgramData\Anaconda3\python.exe)*   

**4. Run the snappy-conf script to configure SNAP**  
If you use the base environemnt python or you already have SNAP installed then,  
Open a command prompt, navigate to SNAP’s bin folder, and run:  
```bash
cd "C:\Program Files\esa-snap\bin"   
snappy-conf "C:\Users\YourUsername\.conda\envs\snap_env\python.exe"
```  
You should see: *Configuration finished successfully!*

**4. Verify esa-snappy works**  
Activate your environment and open Python:  
```bash
conda activate snap_env  
python
```
in the Python environment, try importing:  
```bash
import esa_snappy 
from esa_snappy import ProductIO 
```
If no errors occur, your setup is complete! 

## Documentation

For detailed documentation and examples, see the [documentation website](https://rabinatwayana.github.io/pysarflow/).

## Examples

Check out the 'examples' directory for more examples:

## Contributing

Contributions are welcome! Follow [dev setup guide](./docs/dev.md) & Please feel free to submit a Pull Request.

## Acknowledgments

- This library is build as a part of python software development course at Paris Lodron University Salzburg
- Built on top of powerful open-source libraries like numpy, rasterio

## Contributors

[![Contributors](https://contrib.rocks/image?repo=rabinatwayana/pysarflow)](https://github.com/rabinatwayana/pysarflow/graphs/contributors)
