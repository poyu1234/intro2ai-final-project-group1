import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import string

def generate_random_word(min_len=2, max_len=6):
    """Generates a random word-like string of uppercase letters."""
    length = random.randint(min_len, max_len)
    return ''.join(random.choice(string.ascii_uppercase) for _ in range(length))

def generate_images(num_samples, N, image_size, add_text=True,line_thickness_range = (1, 5)) -> tuple[list[Image.Image], list[Image.Image]]: # Added add_text parameter
    width = height = image_size
    original_images = []
    noisy_images = []

    # Sample texts to use in images (can still be used or mixed with random strings)
    sample_texts = ["Hello", "World", "AI", "ML", "Data", "Python", "Image", "Text", "CNN", "GAN", 
                   "Neural", "Network", "Deep", "Learning", "Computer", "Vision", "NLP",
                   "Algorithm", "Model", "Training", "Test", "Code", "Script", "Project",
                   "Analysis", "Pattern", "Recognition", "Feature", "Extract", "Classify",
                   "Segment", "Object", "Detect", "Generate", "Create", "Art", "Design",
                   "Pixel", "Filter", "Layer", "Epoch", "Batch", "Size", "Tensor", "Flow"]

    for _ in range(num_samples):
        original_image = Image.new("RGB", (width, height), "white")
        noisy_image = Image.new("RGB", (width, height), "white")
        draw_original = ImageDraw.Draw(original_image)
        draw_noisy = ImageDraw.Draw(noisy_image)

        rectangles = [(random.randint(10, 20), random.randint(10, 20), width - random.randint(10, 20), height - random.randint(10, 20))]
        for _ in range(N):
            rect = random.choice(rectangles)
            rectangles.remove(rect)
            x1, y1, x2, y2 = rect

            if random.choice([True, False]) and (x2 - x1 > 100):
                split = random.randint(x1 + 50, x2 - 50)
                rectangles.append((x1, y1, split, y2))
                rectangles.append((split, y1, x2, y2))
            elif y2 - y1 > 100:
                split = random.randint(y1 + 50, y2 - 50)
                rectangles.append((x1, y1, x2, split))
                rectangles.append((x1, split, x2, y2))
            else:
                rectangles.append(rect)

        for rect in rectangles:
            x1, y1, x2, y2 = rect
            line_thickness = random.randint(*line_thickness_range)  # Random line thickness within specified range

            draw_original.rectangle([x1, y1, x2, y2], outline="black", width=line_thickness)

            # Add text to rectangles if they're large enough
            rect_width = x2 - x1
            rect_height = y2 - y1
            
            # Only add text if add_text is True and rectangles are large enough and with a certain probability
            if add_text and rect_width > 60 and rect_height > 40 and random.random() < 0.7: # Added add_text condition
                # Calculate safe area inside rectangle to avoid overlapping lines
                padding = line_thickness + 5 # Padding from the drawn rectangle line
                text_safe_x1 = x1 + padding
                text_safe_y1 = y1 + padding
                text_safe_x2 = x2 - padding
                text_safe_y2 = y2 - padding
                
                # Skip if safe area is too small
                if text_safe_x2 - text_safe_x1 < 20 or text_safe_y2 - text_safe_y1 < 15:
                    pass 
                else:
                    # Generate more text: 1 to 3 segments (random words or from sample_texts)
                    num_segments = random.randint(1, 3)
                    text_segments = []
                    for _ in range(num_segments):
                        if random.random() < 0.5: # 50% chance to use a predefined word
                            text_segments.append(random.choice(sample_texts))
                        else: # 50% chance to generate a random word
                            text_segments.append(generate_random_word())
                    
                    text_to_draw = " ".join(text_segments)
                    
                    # Dynamically adjust font size based on available space and text length
                    max_font_h = text_safe_y2 - text_safe_y1
                    # Estimate char width; 0.6 is a rough aspect ratio, + num_segments for spaces
                    # Ensure len(text_to_draw) is not zero to avoid division by zero
                    if len(text_to_draw) == 0:
                        continue # Skip if no text generated

                    char_width_estimator = 0.6 if len(text_to_draw) < 15 else 0.5
                    max_font_w_derived = (text_safe_x2 - text_safe_x1) / (len(text_to_draw) * char_width_estimator + 1)
                    
                    font_size = random.randint(8, max(10, int(min(max_font_h, max_font_w_derived, 25))))
                    if font_size <=0: # Ensure font size is positive
                        font_size = 8 

                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except IOError:
                        font = ImageFont.load_default() # Fallback to default font

                    # Calculate text dimensions using getbbox or getsize
                    if hasattr(font, "getbbox"): # Pillow >= 9.2.0
                        text_bbox = font.getbbox(text_to_draw)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                    elif hasattr(font, "getsize"): # Older Pillow versions
                        text_width, text_height = font.getsize(text_to_draw)
                    else: # Fallback if neither is found (should not happen with standard Pillow)
                        text_width = len(text_to_draw) * (font_size // 2)
                        text_height = font_size
                    
                    # Ensure text fits, if not, skip or try smaller font (here, we just check)
                    if text_width <= (text_safe_x2 - text_safe_x1) and text_height <= (text_safe_y2 - text_safe_y1):
                        
                        if text_safe_x1 >= (text_safe_x2 - text_width): # If text barely fits or is wider
                            text_x = text_safe_x1
                        else:
                            text_x = random.randint(text_safe_x1, int(text_safe_x2 - text_width))
                        
                        if text_safe_y1 >= (text_safe_y2 - text_height): # If text barely fits or is taller
                            text_y = text_safe_y1
                        else:
                            text_y = random.randint(text_safe_y1, int(text_safe_y2 - text_height))
                        
                        # Draw text on original image
                        draw_original.text((text_x, text_y), text_to_draw, fill="black", font=font)

                        noisy_text_x_offset = random.randint(-2, 2)
                        noisy_text_y_offset = random.randint(-2, 2)

                        noisy_text_x = text_x + noisy_text_x_offset
                        noisy_text_y = text_y + noisy_text_y_offset
                        
                        draw_noisy.text((noisy_text_x, noisy_text_y), text_to_draw, fill="black", font=font)

            x1_noisy = x1
            y1_noisy = y1
            x2_noisy = x2
            y2_noisy = y2

            random_int = lambda: random.randint(-3, 3)

            def draw_wobbly_line(draw_surface, p1, p2, fill_color, line_width, segments=5, wobble_amount=3):
                """Draws a wobbly line between two points."""
                (x1, y1) = p1
                (x2, y2) = p2
                dx = (x2 - x1) / segments
                dy = (y2 - y1) / segments
                
                current_x, current_y = x1, y1
                for i in range(segments):
                    next_x_ideal = x1 + (i + 1) * dx
                    next_y_ideal = y1 + (i + 1) * dy
                    
                    wobble_x = random.randint(-wobble_amount, wobble_amount)
                    wobble_y = random.randint(-wobble_amount, wobble_amount)
                    
                    end_segment_x = next_x_ideal + wobble_x
                    end_segment_y = next_y_ideal + wobble_y

                    # Ensure the final point is exactly p2
                    if i == segments - 1:
                        end_segment_x = x2
                        end_segment_y = y2
                        
                    draw_surface.line([(current_x, current_y), (end_segment_x, end_segment_y)], fill=fill_color, width=line_width)
                    current_x, current_y = end_segment_x, end_segment_y

            if x1 == x2: # Vertical line case - adjust wobble for horizontal segments
                # Top horizontal segment (potentially wobbly)
                draw_wobbly_line(draw_noisy, (x1_noisy + random_int(), y1_noisy), (x2_noisy + random_int(), y1_noisy), "black", line_thickness)
                # Right vertical segment
                draw_wobbly_line(draw_noisy, (x2_noisy + random_int(), y1_noisy), (x2_noisy + random_int(), y2_noisy), "black", line_thickness)
                # Bottom horizontal segment (potentially wobbly)
                draw_wobbly_line(draw_noisy, (x2_noisy + random_int(), y2_noisy), (x1_noisy + random_int(), y2_noisy), "black", line_thickness)
                # Left vertical segment
                draw_wobbly_line(draw_noisy, (x1_noisy + random_int(), y2_noisy), (x1_noisy + random_int(), y1_noisy), "black", line_thickness)
            else: # Horizontal line case - adjust wobble for vertical segments
                # Top horizontal line
                draw_wobbly_line(draw_noisy, (x1_noisy, y1_noisy + random_int()), (x2_noisy, y1_noisy + random_int()), "black", line_thickness)
                # Right vertical segment (potentially wobbly)
                draw_wobbly_line(draw_noisy, (x2_noisy, y1_noisy + random_int()), (x2_noisy, y2_noisy + random_int()), "black", line_thickness)
                # Bottom horizontal line
                draw_wobbly_line(draw_noisy, (x2_noisy, y2_noisy + random_int()), (x1_noisy, y2_noisy + random_int()), "black", line_thickness)
                # Left vertical segment (potentially wobbly)
                draw_wobbly_line(draw_noisy, (x1_noisy, y2_noisy + random_int()), (x1_noisy, y1_noisy + random_int()), "black", line_thickness)

        original_image = original_image.convert('L')
        noisy_image = noisy_image.convert('L')

        original_images.append(original_image)
        noisy_images.append(noisy_image)

    return original_images, noisy_images