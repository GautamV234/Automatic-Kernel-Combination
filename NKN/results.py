from PIL import Image

# Load the three images
img1 = Image.open('/home/gautam.pv/nlim/NKN/outputs/cheetahs/manual.png')
img2 = Image.open('/home/gautam.pv/nlim/NKN/outputs/cheetahs/semi_grid.png')
img3 = Image.open('/home/gautam.pv/nlim/NKN/outputs/cheetahs/neural.png')

# Get the width and height of the images
width, height = img1.size

# Create a new image with the required dimensions
result_img = Image.new('RGBA', (width, height * 3))

# Paste the three images onto the new image
result_img.paste(img1, (0, 0))
result_img.paste(img2, (0, height))
result_img.paste(img3, (0, height * 2))

# Save the result image
result_img.save('result.png')