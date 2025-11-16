from resize512 import resize_image_to_512x512_and_convert
from draw_cartoon_df import process_folder_to_cartoon
from cropx4 import crop512
from combine import stitch_2x2_to_512
from remove_bg import rmbg_image

def dfcall(input_file):
    input_file= input_file
    rmbg_image(input_file)
    input_file = r"output_with_white_bg.png" # ➡️ เปลี่ยนเป็นชื่อไฟล์รูปภาพของคุณ
    output_file = "my_resized_512x512_image.jpg" # ➡️ เปลี่ยนเป็นชื่อไฟล์ผลลัพธ์ที่คุณต้องการ
    resize_image_to_512x512_and_convert(input_file,output_file)
    image = r"my_resized_512x512_image.jpg"
    output_folder = "cropped_parts"
    crop512(image,output_folder)
    input_folder = "cropped_parts"  # โฟลเดอร์ที่มีรูป 256x256 ที่ถูก crop
    output_folder = "cartoon_output" # โฟลเดอร์สำหรับผลลัพธ์
    process_folder_to_cartoon(input_folder,output_folder)
    input_folde = "cartoon_output"
    stitch_2x2_to_512(input_folde)

image=r"/Users/student/Desktop/DobotDrawRESEARCH/IMG_20250805_151301.jpg"
dfcall(image)