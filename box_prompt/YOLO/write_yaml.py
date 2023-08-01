yaml_content = """
path: '/root/autodl-tmp/Medical_SAM/data/TNBC_CryoNuSeg_MoNuSeg/'
train: 'train/images'
val: 'valid/images'

# class names
names: 
  0: 'nuclei'
"""

with open('TNBC_CryoNuSeg_MoNuSeg.yaml', 'w') as yaml_file:
    yaml_file.write(yaml_content)
