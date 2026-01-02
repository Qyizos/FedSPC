import os
import imageio
from PIL import Image
from utils.my_transforms import get_transforms

def getVal(client = 'TNBC', type='val'):
    transform_test = {
        'to_tensor': 1
    }

    # data transforms
    test_transform = get_transforms(transform_test)
    dir_aa = './data_for_train/{:s}/images/{:s}'.format(client, type)
    testPath = client.split('-')[0]
    dir_cc = './data/{:s}/labels_instance'.format(testPath)

    aa_files = [f for f in os.listdir(dir_aa) if f.endswith('.png')]
    data_list = []

    for aa_file in aa_files:
        file_name_without_extension = os.path.splitext(aa_file)[0]
        cc_file_with_label = os.path.join(dir_cc, f"{file_name_without_extension}_label.png")

        aa_file_full = os.path.join(dir_aa, aa_file)

        if os.path.exists(cc_file_with_label):
            input = Image.open(aa_file_full)
            valLabel = imageio.imread(cc_file_with_label)
            input = test_transform((input,))[0].unsqueeze(0)
            data_list.append((input, valLabel))

    return data_list

if __name__=='__main__':
    getVal()


