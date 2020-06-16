from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import pandas as pd
import SimpleITK as sitk
from Preprocessing_utlities import extract_class_balanced_example_array
from Preprocessing_utlities import resize_image_with_crop_or_pad
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from config import*


########################-------Fucntions for tf records
# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def flow_from_df(dataframe: pd.DataFrame, chunk_size):
    for start_row in range(0, dataframe.shape[0], chunk_size):
        end_row  = min(start_row + chunk_size, dataframe.shape[0])
        yield dataframe.iloc[start_row:end_row, :]


def creat_tfrecord(df,extraction_perameter,tf_name):

    read_csv=df.as_matrix()
    patch_params = extraction_perameter

    img_list=[]
    mask_list=[]
    lbl_list=[]
    id_name=[]

    for Data in read_csv:
        img_path = Data[4]
        subject_id = img_path.split('/')[-1].split('.')[0]
        Subject_lbl=Data[5:10]
        print(Subject_lbl.shape)

        print('Subject ID-{}'.format(subject_id))
        print('Labels--{}'.format(Subject_lbl))

        #Img
        img_sitk = sitk.ReadImage(img_path, sitk.sitkFloat32)
        image= sitk.GetArrayFromImage(img_sitk)
        #Mask
        mask_fn = str(Data[10])
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_fn)).astype(np.int32)
        print(mask.shape)
        print(image.shape)

        patch_size=patch_params['example_size']
        img_shape=image.shape

        ###----padding_data_if_needed
       #####----z dimention-----######
        if (patch_size[0] >=img_shape[0]):
            dimention1=patch_size[0]+10
        else:
            dimention1=img_shape[0]

        #####----x dimention-----######
        if (patch_size[1] >=img_shape[1]):
             dimention2=patch_size[1]+10
        else:
            dimention2=img_shape[1]

        #####----Y dimention-----######
        if (patch_size[2] >=img_shape[2]):
             dimention3=patch_size[2]+10
        else:
            dimention3=img_shape[2]
        print('------before padding image shape--{}-----'.format(image.shape))
        image=resize_image_with_crop_or_pad(image, [dimention1,dimention2,dimention3], mode='symmetric')
        mask=resize_image_with_crop_or_pad(mask, [dimention1,dimention2,dimention3], mode='symmetric')
        print('######before padding image shape--{}#####'.format(image.shape))



        img_shape=image.shape
        image= np.expand_dims(image, axis=3)

        images,masks = extract_class_balanced_example_array(
                    image,mask,
                    example_size=patch_params['example_size'],
                    n_examples=patch_params['n_examples'],
                    classes=4,class_weights=[0,0,1,1])

        print(images.shape)

        for e in range(patch_params['n_examples']):
            img_list.append(images[e][:,:,:,0])
            #print(images[e][:,:,:,0].shape)
            mask_list.append(masks[e][:,:,:])
            #print('Mask-Shape=={}'.format(masks[e][:,:,:].shape))
            lbl_list.append(Subject_lbl)
            patch_name=str(subject_id+'_{}'.format(e))
            #Converting_string_bytes
            patch_name =bytes(patch_name, 'utf-8')
            #print(patch_name)
            id_name.append(patch_name)

    print('This Rfrecords will contain--{}--Pathes--of-size--{}'.format(len(id_name),patch_params['example_size']))

    record_mask_file = tf_name
    with tf.io.TFRecordWriter(record_mask_file) as writer:
         for e in range(len(img_list)):
            feature = {'label1': _int64_feature(lbl_list[e][0]),
                       'label2': _int64_feature(lbl_list[e][1]),
                       'label3': _int64_feature(lbl_list[e][2]),
                       'label4': _int64_feature(lbl_list[e][3]),
                       'label5': _int64_feature(lbl_list[e][4]),
                        'image':_bytes_feature(img_list[e].tostring()),
                        'mask':_bytes_feature(mask_list[e].tostring()),
                        'Height':_int64_feature(patch_params['example_size'][0]),
                        'Weight':_int64_feature(patch_params['example_size'][1]),
                        'Depth':_int64_feature(patch_params['example_size'][2]),
                        'label_shape':_int64_feature(5),
                        'Sub_id':_bytes_feature(id_name[e])
                        }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    writer.close()

    return


@tf.function
def decode_ct(Serialized_example):

    features={
       'label1': tf.io.FixedLenFeature([],tf.int64),
        'label2': tf.io.FixedLenFeature([],tf.int64),
        'label3': tf.io.FixedLenFeature([],tf.int64),
        'label4': tf.io.FixedLenFeature([],tf.int64),
        'label5': tf.io.FixedLenFeature([],tf.int64),
       'image':tf.io.FixedLenFeature([],tf.string),
       'mask':tf.io.FixedLenFeature([],tf.string),
       'Height':tf.io.FixedLenFeature([],tf.int64),
       'Weight':tf.io.FixedLenFeature([],tf.int64),
       'Depth':tf.io.FixedLenFeature([],tf.int64),
       'label_shape':tf.io.FixedLenFeature([],tf.int64),
        'Sub_id':tf.io.FixedLenFeature([],tf.string)

     }
    examples=tf.io.parse_single_example(Serialized_example,features)
    ##Decode_image_float
    image_1 = tf.io.decode_raw(examples['image'], float)
    #Decode_mask_as_int32
    #mask_1 = tf.io.decode_raw(examples['mask'], tf.int32)
    ##Subject id is already in bytes format
    #sub_id=examples['Sub_id']


    img_shape=[examples['Height'],examples['Weight'],examples['Depth']]
    #img_shape2=[img_shape[0],img_shape[1],img_shape[2]]
    print(img_shape)
    #Resgapping_the_data
    img=tf.reshape(image_1,img_shape)
    #Because CNN expect(batch,H,W,D,CHANNEL)
    img=tf.expand_dims(img, axis=-1)
    #mask=tf.reshape(mask_1,img_shape)
    #mask=tf.expand_dims(mask, axis=-1)
    ###casting_values
    img=tf.cast(img, tf.float32)
    #mask=tf.cast(mask,tf.int32)

    lbl=[examples['label1'],examples['label2'],examples['label3'],examples['label4'],examples['label5']]
    ##Transpossing the Multilabels
    #lbl=tf.linalg.matrix_transpose(lbl)
    return img,lbl
