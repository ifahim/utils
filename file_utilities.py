# Move files from one directory to another
import shutil, os
def move_file_from_one_directory_to_another(src_dir, tgt_dir):
    files_to_move = os.listdir(src_dir)
    print('Number of files to move: ', len(files_to_move))
    i = 0
    for afile in files_to_move:
        i += 1
        if i % 10000 == 0:
            print(i)
        src_file = os.path.join(src_dir, afile)
        tgt_file = os.path.join(tgt_dir, afile)
        shutil.move(src_file, tgt_file)

#Use it like this
# src_dir = './labels/'
# tgt_dir = './images/'
# move_file_from_one_directory_to_another(src_dir, tgt_dir)