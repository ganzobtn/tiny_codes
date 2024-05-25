import cv2
import os
import numpy as np
import pickle
import argparse
from scipy.special import softmax
import pandas as pd
from multiprocessing import Pool
from get_predictions import merge, get_preds
import multiprocessing as mp
def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Confusion Video')
    parser.add_argument('--path', type=str, default='/projects/results/videomaev2/finetune/test/aslcitizen_2731/vit_b_32_asl_citizen_head_hands_square_resized_merged_ft_from_wlasl_2000/1',
                        help='Path to the directory containing the results')
    parser.add_argument('--video_dir', type=str, default='/projects/data/asl-citizen/ASL_Citizen',
                        help='Path to the directory containing the videos')
    parser.add_argument('--save_dir', type=str, default='/projects/results/videomaev2/finetune/predictions/vit_b_32_asl_citizen_head_hands_square_resized_merged_ft_from_wlasl_2000_original_videos/confusion_videos',
                        help='Path to the directory to save the confusion videos')
    parser.add_argument('--top_k', type=int, default=1,
                        help='Number of top predictions to consider')
    parser.add_argument('--labels_path', type=str, default='/projects/videomaev2/datas/dgx/finetune/revised/24.04.29/aslcitizen_2731',
                        help='Name of the file containing the output labels')
    parser.add_argument('--output_logits_file', type=str, default='0.txt',
                        help='Name of the file containing the output logits')
    parser.add_argument('--output_names_file', type=str, default='output_names_test.pkl',
                        help='Name of the file containing the output names')
    parser.add_argument('--vocab_file', type=str, default='/projects/data/asl-citizen/ASL_Citizen/vocab.txt',
                        help='Name of the file containing the vocabulary')
    parser.add_argument('--num_tasks', type=int, default=8)

    return parser.parse_args()


def read_output_logits(path,num_tasks=1):
    #output_labels = [0,1,2,3,4,5]
    #output_logits = np.array([[0.5,0.3,0.2],[0.2,0.1,0.7],[0.1,0.2,0.7],[0.1,0.2,0.7],[0.1,0.2,0.7],[0.1,0.2,0.7]])
    input_lst = merge(path, num_tasks, method='prob')
    p = Pool(64)
    preds = p.map(get_preds,input_lst)
    p.close()
    video_ids = [i[0] for i in preds]
    output_logits = [i[1] for i in preds]
    output_labels = [i[2] for i in preds]

    #output_labels  = np.load(os.path.join(path,'labels_test.npy'))
    #output_logits = np.load(os.path.join(path,'gloss_logits_test.npy'))
    
    return video_ids ,output_logits,output_labels


def read_concat_videos(videos_label, pred_videos_label,video_num,data_root,video_height,video_width):
    n_columns = {'train': 6, 'test':6, 'val': 2}
    max_frames = 0
    videos_array_whole = []
    n_rows = []
    for videos in [videos_label, pred_videos_label]:
        n_rows.append(max([int(np.ceil(len(videos[i])/n_columns[i] ))  for i in n_columns.keys()]))
        
        #print(n_rows)
        videos_array = {}
        for split in ['train','test','val']:
            #print(len(videos[split]))
            array_list = []
            for video in videos[split]:

                cap = cv2.VideoCapture(os.path.join(data_root,video))
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                #video_array = np.zeros(length,3,256,256)
                video_array = []
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # Read until video is completed
                while(cap.isOpened()):
                # Capture frame-by-frame
                    ret, frame = cap.read()
                    if ret == True:
                        
                        # Display the resulting frame
                        #cv2.imshow('Frame',frame)
                        if video_height != height or video_width != width:
                            frame = cv2.resize(frame, (video_width, video_height))
                        video_array.append(frame)
                    
                    # Break the loop
                    else: 
                        break
                #print(len(video_array),video_array[0].shape)
                if max_frames<len(video_array):
                    max_frames = len(video_array)
                video_array = np.array(video_array)
                #print(video_array.shape)
                array_list.append(video_array)
                # When everything done, release the video capture object
                cap.release()
            
            videos_array[split] = array_list
        videos_array_whole.append(videos_array)
    return videos_array_whole, max_frames,n_columns,n_rows,video_num


def save_video(list_videos_array_whole,n_columns,n_rows,classes,max_frames,video_num,save_path,video_height,video_width,video_id=''):
    videos_array_shape = ( video_width*sum(n_columns.values()), video_height* sum(n_rows))
    
    gt_class =  classes[0]
    pred_class = classes[1]
    
    # save_path = '/projects/data/wlasl_2000/predictions/wlasl_1522_two_frame64_boston/confusion_videos'
    out = cv2.VideoWriter(os.path.join(save_path,gt_class+'_'+pred_class+'_'+video_id+'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 30, videos_array_shape)

    grid_size = (sum(n_rows) , sum(n_columns.values()))
    column_starts= {'train':0,'test':n_columns['train'],'val':n_columns['train']+n_columns['test']}
    row_starts = [0] + list(np.cumsum(n_rows)[:-1])

    for i in range(max_frames):

        frame_img = np.zeros((video_height* grid_size[0], video_width* grid_size[1], 3 ),dtype= np.uint8)

        for j in range(len(n_rows)):
            for split in ['train', 'test', 'val']:

                for k, array in enumerate(list_videos_array_whole[j][split]):  # n_videos,  (n_framesx256x256x3)

                    grid_number = (row_starts[j] + k//n_columns[split],column_starts[split]+k%n_columns[split])
                    
                    frame_number = min(array.shape[0],i)-1
                    frame_img[video_height*grid_number[0]:video_height*(grid_number[0]+1),video_width*grid_number[1]:video_width*(grid_number[1]+1)]= array[frame_number]

                    if split=='test' and k==video_num and j==0:
                        start_point = (video_width*grid_number[1], video_height*grid_number[0])
                        end_point = (video_width*(grid_number[1]+1), video_height*(grid_number[0]+1))
                        
                        color = (0,0,255)
                        thickness = 6
                        frame_img = cv2.rectangle(frame_img, start_point, end_point, color, thickness) 

        # Draw seperating line between train,test,val, gtruth,preds
        # cv2.line(image, start_point, end_point, color, thickness)
        for cls_num,j in enumerate([0]+list(np.cumsum(n_rows)[:-1])):
                # font 
                font = cv2.FONT_HERSHEY_SIMPLEX 
                # org 
                #org = (256*j+10, 10) 
                org = (10, video_height*j+30) 
                 
                # fontScale 
                fontScale = 1
                # Blue color in BGR 
                color = (255, 255, 0) 
                # Line thickness of 2 px 
                thickness = 6              

                frame_img= cv2.putText(frame_img, classes[cls_num], org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 

        for cls_num,j in enumerate( np.cumsum(n_rows)[:-1]):
                # font 
                font = cv2.FONT_HERSHEY_SIMPLEX 
                # org 
                #org = (256*j+10, 10) 
                org = (10, video_height*j+10) 
                 
                # fontScale 
                fontScale = 1
                # Blue color in BGR 
                color = (255, 255, 0) 
                # Line thickness of 2 px 
                thickness = 6              
                # start_point = (256*j,0)
                # end_point =  (256*j,256*sum(n_columns.values()))
                start_point = (0,video_height*j)
                end_point =  (video_width*sum(n_columns.values())-1, video_height*j)

                cv2.line(frame_img, start_point, end_point, color=(255,0,0),thickness=thickness)

        for k in np.cumsum(list(n_columns.values()))[:-1]:
                start_point = (video_width*k, 0)
                end_point =  ( video_width*k, video_height*sum(n_rows)-1)
                #print('start_end:',start_point, end_point)
                frame_img = cv2.line(frame_img, start_point, end_point , color=(0,255,0),thickness=6)

        out.write(frame_img)

    out.release()

args = parse_args()
path = args.path
video_dir = args.video_dir
save_dir = args.save_dir
top_k = args.top_k
labels_path = args.labels_path
output_logits_file = args.output_logits_file
output_names_file = args.output_names_file
vocab_file = args.vocab_file
num_tasks = args.num_tasks
# path = '/projects/results/videomaev2/finetune/test/aslcitizen_2731/vit_b_32_asl_citizen_ft_from_wlasl_2000/1'
# video_dir = '/projects/data/asl-citizen/ASL_Citizen'
# save_dir = '/projects/results/videomaev2/finetune/predictions/vit_b_32_asl_citizen_ft_from_wlasl_2000/confusion_videos'
# top_k = 1
# labels_path = '/projects/videomaev2/datas/dgx/finetune/revised/24.04.29/aslcitizen_2731'
# output_logits_file = '0.txt'
# #output_names_file = args.output_names_file
# vocab_file = '/projects/data/asl-citizen/ASL_Citizen/vocab.txt'
# num_tasks = 8
os.makedirs(save_dir,exist_ok=True)

video_ids, output_logits, output_labels = read_output_logits(path,num_tasks)

# with open(os.path.join(path,output_names_file), 'rb') as handle:
#     output_names = pickle.load(handle)

with open(vocab_file) as file_in:
    vocab = []
    for line in file_in:
        vocab.append(line.lower().replace('\n',''))

file_dictionary = {}
for mode in ['train','test','val']:
    file = pd.read_csv(os.path.join(labels_path,mode+'.csv'), header=None, delimiter=' ')
    
    file_dictionary[mode] = {i:[]    for i in vocab}

    for i in range(len(file)):
    #    a = file.iloc[i,1]
    #    b = file.iloc[i,0]
       file_dictionary[mode][vocab[file.iloc[i,1]]].append(file.iloc[i,0])    


preds = np.argmax(output_logits,axis= 1)
top_k = 1

a = np.argsort(output_logits,axis=1)[:,-top_k:]
predictions = []
for i in range(len(output_labels)):
    predictions.append([vocab[k] for k in a[i]])


def process_prediction(args):
    i, pred  = args
    count = 0
    try:
        if pred != output_labels[i]:
            label_videos_path = {mode: file_dictionary[mode][vocab[output_labels[i]]] for mode in ['train', 'test', 'val']}
            pred_videos_path = {mode: file_dictionary[mode][vocab[pred]] for mode in ['train', 'test', 'val']}
            video_num = next((idx for idx, video in enumerate(label_videos_path['test']) if video_ids[i] in video), None)
            if video_num is None:
                print('error:', i, output_labels[i])
                return 0
            list_videos_array_whole, max_frames, n_columns, n_rows, video_num = read_concat_videos(
                label_videos_path, pred_videos_path, video_num, data_root=video_dir, video_height=240, video_width=320
            )
            pred_class = predictions[i][-1]
            gt_class = vocab[output_labels[i]]
            classes = [vocab[output_labels[i]]] + [j for j in reversed(predictions[i])]
            print(pred_class, gt_class, classes)
            save_video(list_videos_array_whole, n_columns, n_rows, classes, max_frames, video_num,
                       save_path=save_dir, video_height=240, video_width=320, video_id=video_ids[i])
            count += 1
    except Exception as e:
        print('error:', i, output_labels[i], str(e))
        pass
    return count

# Create a list of arguments for each prediction
args_list = [(i, preds[i]) for i in range(len(preds))]

with mp.Pool(mp.cpu_count()) as pool:
    results = pool.map(process_prediction, args_list)

total_count = sum(results)
print(f'Total mismatches processed: {total_count}')
#  count = 0
# for i, pred in enumerate(preds):
#     if pred!=output_labels[i]:
#         count +=1
#         # if True:
#         try:

#             label_videos_path = {mode: file_dictionary[mode][vocab[output_labels[i]]] for mode in ['train','test','val']}
#             pred_videos_path = {mode: file_dictionary[mode][vocab[pred]] for mode in ['train','test','val']}

#             video_num = next((idx for idx, video in enumerate(label_videos_path['test']) if video_ids[i] in video), None)
#             if video_num is None:
#                 print('error:',i,output_labels[i])
#                 continue
#             list_videos_array_whole, max_frames ,n_columns, n_rows, video_num  = read_concat_videos(label_videos_path, pred_videos_path,video_num,data_root=video_dir,video_height=240,video_width=320)

#             pred_class = predictions[i][-1]
#             gt_class = vocab[output_labels[i]]
#             classes =  [vocab[output_labels[i]]]+[j for j in reversed(predictions[i])]
#             print(pred_class,gt_class,classes)  
#             save_video(list_videos_array_whole,n_columns,n_rows,classes,max_frames,video_num,save_path =save_dir , video_height=240,video_width=320,video_id=video_ids[i])

#         except:
#             print('error:',i,output_labels[i])
#             # continue
#             pass
# print(count)