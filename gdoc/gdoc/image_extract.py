from PIL import Image

def resolution():
    pic = Image.open("/opt/hyrin/frappe-bench/apps/gdoc/gdoc/gdoc/docs/ocr_image.png")
    res = pic.size
    print(res)


resolution()