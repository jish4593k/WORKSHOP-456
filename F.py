import os
import asyncio
from pyppeteer import launch
from torchvision import transforms
from tkinter import Tk, Label, Button, filedialog
from PIL import Image
import torch
from scipy.stats import zscore
import numpy as np

async def get_slides_ss(uname, email, passwd, deck_name, ss_dir):
    browser = await launch(headless=True)
    page = await browser.newPage()

    await page.goto('https:///sign_in')
    await page.type('#user_email', emil)
    await page.type('#user_psswd')
    await page.click('button')

    await page.waitForSelector('.picture')
    await page.goto(f'httpam#/')

    await page.waitForSelector('footer button')
    await asyncio.sleep(2)
    await page.click('footer button')

    ss_idx = 0
    while True:
        await asyncio.sleep(2)
        ss_fname = f'{ss_dir}/{SCREENSHOT_BASENAME}{ss_idx:04d}.png'
        await page.screenshot({'path': ss_fname})

        right_arrow = await page.waitForXPath('//button[2]')
        down_arrow = await page.waitForXPath('//button[4]')

        if 'enabled' in await down_arrow.getProperty('className').jsonValue():
            await down_arrow.click()
            print(f'Wrote: {ss_fname}')
            ss_idx += 1
        elif 'enabled' in await right_arrow.getProperty('className').jsonValue():
            await right_arrow.click()
            print(f'Wrote: {ss_fname}')
            ss_idx += 1
        else:
            break

    await browser.close()
    return ss_idx

def make_ss_dir(ss_dir):
    if os.path.exists(ss_dir):
        for old_file in os.listdir(ss_dir):
            old_path = os.path.join(ss_dir, old_file)
            if os.path.isfile(old_path):
                os.remove(old_path)
    else:
        os.makedirs(ss_dir)

def convert_to_pdf(ss_dir, pdf_fname):
    ss_files = [f'{ss_dir}/{SCREENSHOT_BASENAME}{i:04d}.png' for i in range(num_ss)]
    images = [Image.open(img_path) for img_path in ss_files]

  
    transform = transforms.Compose([
        transforms.Resize((300, 400)),
        transforms.ToTensor(),
    ])

    processed_images = [transform(img) for img in images]
    stacked_images = torch.stack(processed_images)

  
    flattened_images = stacked_images.view(num_ss, -1)
    z_scores = zscore(flattened_images.numpy(), axis=None)
    normalized_images = torch.tensor(z_scores).view(num_ss, *stacked_images.shape[1:])

   
    processed_image_paths = [f'{ss_dir}/processed_{i:04d}.png' for i in range(num_ss)]
    for i, img in enumerate(normalized_images):
        img = transforms.ToPILImage()(img)
        img.save(processed_image_paths[i])

    
    pdf_file_path = f'{ss_dir}/{pdf_fname}'
    images_array = np.array([np.array(Image.open(img_path)) for img_path in processed_image_paths])
    scipy.misc.imsave(pdf_file_path, convert(images_array, dpi=300, fmt='pdf'))



if __name__ == '__main__':
    main()
