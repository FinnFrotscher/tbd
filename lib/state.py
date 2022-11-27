import os, datetime, math, time, cv2
from os import path
from tinydb import TinyDB

root_path = path.join(os.getcwd())
out_path = path.normpath(path.join(root_path, '..',  'output/'))
db_path = path.normpath(path.join(out_path, 'db.json'))

db = TinyDB(db_path)
cwd = path.join(os.getcwd())


    # img_path = f"/home/finn/code/tbd/tbd/output/tmp-{name}.png"
    # output = climage.convert(img_path, width=80, is_unicode=True)
    # print(output)
    # os.remove(img_path)

    # cv2.imshow("test2", frame)
    # cv2.namedWindow("test2")

def save_loop(index = None, prompt = None, primer = None, image = None ):
    timestamp = datetime.datetime.now().isoformat()

    # image_path = out_path + f"images/{index}_image_{timestamp}.png"
    primer_path = path.join(out_path, "primer", f"{timestamp}_primer_{index}.png")

    print(index, primer_path, primer.shape, prompt)

    # cv2.imwrite(image_path, image)
    cv2.imwrite(primer_path, primer)

    db.insert({
        'index': index,
        'timestamp': timestamp,
        'prompt': prompt,
        'primer_path': primer_path,
        # 'image_path': image_path,
    })

