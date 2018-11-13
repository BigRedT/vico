import os
import shutil
from multiprocessing import Pool
from itertools import product
import subprocess
from PIL import Image
import urllib.request


import utils.io as io
from .constants import ImagenetConstants


def construct_wget_cmd(url,name,directory_prefix):
    cmd = [
        'wget',
        '-q',
        '-O',
        f'{directory_prefix}/{name}',
        url]
    return cmd


def scale(img,smaller_side):
    img_w, img_h = img.size

    if img_w > img_h:
        img_w = int(img_w * smaller_side / float(img_h))
        img_h = smaller_side
    else:
        img_h = int(img_h * smaller_side / float(img_w))
        img_w = smaller_side

    img = img.resize((img_w,img_h))
    return img


def downloader(img_dir,urls):
    wnid, subwnid_to_url = urls
    
    print(f'Downloading images for {wnid} ...')

    wnid_img_dir = os.path.join(img_dir,wnid)
    if os.path.exists(wnid_img_dir):
        return

    io.mkdir_if_not_exists(wnid_img_dir)

    count = 0
    for subwnid, url in subwnid_to_url.items():
        if count > 100:
            break
        
        name = f'{subwnid}.jpg'
        cmd = construct_wget_cmd(url,name,wnid_img_dir)

        path = os.path.join(wnid_img_dir,name)

        try:
            process = subprocess.run(cmd,timeout=10)
        except subprocess.TimeoutExpired:
            print('Taking too long')
            continue

        if process.returncode!=0:
            if os.path.exists(path):
                subprocess.run(['rm',path])
            continue

        try:
            img = Image.open(path)
            img = scale(img,256)
            img.save(path)
            count += 1
        except:
            if os.path.exists(path):
                subprocess.run(['rm',path])


def main():
    const = ImagenetConstants()
    io.mkdir_if_not_exists(const.img_dir)

    print('Loading urls ...')
    wnid_to_urls = io.load_json_object(const.wnid_to_urls_json)

    print('Starting pool ...')
    with Pool(40) as p:
        p.starmap(downloader,product([const.img_dir],wnid_to_urls.items()))


if __name__=='__main__':
    main()