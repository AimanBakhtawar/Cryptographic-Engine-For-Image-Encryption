from django.shortcuts import render, redirect
import string, secrets, json, re, os, io, base64, cv2, random, traceback, time, hashlib
from django.http import JsonResponse
from django.core.mail import send_mail
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.contrib import messages
from .models import EncryptedImage, AttackResult
from io import BytesIO
import numpy as np
from django.conf import settings
from django.shortcuts import get_object_or_404
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# Home Page
@login_required
def IndexPage(request):
    return render(request, 'Index.html')

def is_strong_password(password):
    return (
        len(password) >= 8 and
        re.search(r'[A-Z]', password) and
        re.search(r'[a-z]', password) and
        re.search(r'[0-9]', password) and
        re.search(r'[\W_]', password)
    )

# Registration Page
def RegisterPage(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')

        if User.objects.filter(username=uname).exists():
            return render(request, 'Registration.html', {'error': 'Username already exists'})
        
        if not is_strong_password(password):
            return render(request, 'Registration.html', {'error': 'Password must be at least 8 characters long and include uppercase, lowercase, number, and special character.'})

        user = User.objects.create_user(username=uname, email=email, password=password)
        user.save()

        return redirect('login')

    return render(request, 'Registration.html')

# Login Page
def LoginPage(request):
    if request.method == 'POST':
        uemail = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, username=uemail, password=password)
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            # Set an error message
            messages.error(request, 'Invalid user Email or Password')
            return redirect('login')  # Redirect back to the login page
    return render(request, 'Login.html')

# Logout Page
def LogoutPage(request):
    logout(request)
    return redirect('login')

# Secure Random Key
def generate_key(request):
    if request.method == 'GET':
        characters = string.ascii_letters + string.digits + string.punctuation
        key = ''.join(secrets.choice(characters) for _ in range(16))
        return JsonResponse({'key': key})
    return JsonResponse({'error': 'Invalid request'}, status=400)

# Check Key Strength
def is_key_strong(key):
    # start = time.time()
    result = all([
        len(key) >= 10,
        re.search(r'[A-Z]', key),
        re.search(r'[a-z]', key),
        re.search(r'[0-9]', key),
        re.search(r'[\W_]', key)
    ])
    # print(f"[DEBUG] is_key_strong() took {time.time() - start:.4f} seconds")
    return result

# Calculate Key Space
def calculate_key_space(key_length):
    # start = time.time()
    character_set_size = len(string.ascii_letters + string.digits + string.punctuation)
    result = character_set_size ** key_length
    # print(f"[DEBUG] calculate_key_space() took {time.time() - start:.4f} seconds")
    return result

# Split into blocks
def split_into_blocks(image, block_size):
    h, w, c = image.shape
    padded_h = (h + block_size - 1) // block_size * block_size
    padded_w = (w + block_size - 1) // block_size * block_size

    padded_image = np.zeros((padded_h, padded_w, c), dtype=image.dtype)
    padded_image[:h, :w, :] = image

    blocks = []
    for i in range(0, padded_h, block_size):
        for j in range(0, padded_w, block_size):
            block = padded_image[i:i+block_size, j:j+block_size, :]
            blocks.append(((i, j), block))

    return blocks, padded_image.shape

# Arnold Cat Map - Encryption
def generate_chaotic_sequence(image, iterations):
    # start = time.time()
    h, w, c = image.shape
    if h != w:
        raise ValueError("Arnold Cat Map only works for square matrices!")

    N = h
    x, y = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    transformed_image = image.copy()

    for _ in range(iterations):
        new_x = (x + y) % N
        new_y = (x + 2 * y) % N
        transformed_image = transformed_image[new_x, new_y]

    # print(f"[DEBUG] generate_chaotic_sequence() took {time.time() - start:.4f} seconds")
    return transformed_image

# Merge blocks
def merge_blocks(blocks, image_shape):
    reconstructed = np.zeros(image_shape, dtype=np.uint8)
    for (i, j), block in blocks:
        reconstructed[i:i+block.shape[0], j:j+block.shape[1], :] = block
    return reconstructed

# Pixel Shuffle (Encryption)
def shuffle_pixels(image, key_value):
    # start = time.time()
    height, width, channels = image.shape
    total_pixels = height * width

    indices = np.arange(total_pixels)
    np.random.seed(key_value)  
    np.random.shuffle(indices)

    flat_pixels = image.reshape(total_pixels, channels)
    shuffled_pixels = flat_pixels[indices]

    result = shuffled_pixels.reshape(image.shape)
    # print(f"[DEBUG] shuffle_pixels() took {time.time() - start:.4f} seconds")
    return result, indices

# Pixel Value Modifier
def modify_pixel_values(image, key_value):
    start = time.time()
    height, width, channels = image.shape
    modified_image = image.astype(np.uint16)

    np.random.seed(key_value + 99)
    random_values = np.random.randint(1, 256, size=(height, width, channels), dtype=np.uint16)

    modified_image = (modified_image + random_values) % 256
    result = modified_image.astype(np.uint8)
    print(f"[DEBUG] modify_pixel_values() took {time.time() - start:.4f} seconds")
    return result

# Encrypt Image Function
def encrypt_image(image_array, key):
    start_time = time.time()

    key_value = (sum(ord(char) for char in key) * 2)
    key_value = key_value % 100  # keep iterations reasonable

    block_size = 64
    blocks, padded_shape = split_into_blocks(image_array, block_size)

    encrypted_blocks = []
    for (i, j), block in blocks:
        encrypted_block = generate_chaotic_sequence(block, key_value)
        encrypted_blocks.append(((i, j), encrypted_block))

    encrypted_image = merge_blocks(encrypted_blocks, padded_shape)

    shuffle_start = time.time()
    encrypted_image, indices = shuffle_pixels(encrypted_image, key_value)
    print(f"[DEBUG] Shuffle inside encrypt_image() took {time.time() - shuffle_start:.4f} seconds")

    modify_start = time.time()
    encrypted_image = modify_pixel_values(encrypted_image, key_value)
    print(f"[DEBUG] Pixel modification inside encrypt_image() took {time.time() - modify_start:.4f} seconds")

    print(f"[DEBUG] encrypt_image() total execution time: {time.time() - start_time:.4f} seconds")
    return encrypted_image, key_value, indices, image_array.shape

# Main Encryption Page
@login_required
def EncryptionPage(request):
    if request.method == 'POST':
        full_start = time.time()
        image_file = request.FILES.get('image')
        key = request.POST.get('secret-key')

        system_generated = False
        if not key:
            key = generate_key()
            system_generated = True

        if not is_key_strong(key):
            return JsonResponse({'error': 'Weak key! Use uppercase, lowercase, digits, and special characters.'}, status=400)

        if not image_file:
            return JsonResponse({'error': 'No image file selected.'}, status=400)
        image = Image.open(image_file).convert('RGB')
        image_array = np.array(image, dtype=np.uint8)

        encrypted_pixels, key_value, indices, original_size = encrypt_image(image_array, key)
        original_size_str = json.dumps(original_size)

        encrypted_image_pil = Image.fromarray(encrypted_pixels.astype(np.uint8))
        buffer = BytesIO()
        encrypted_image_pil.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)

        encrypted_image_model = EncryptedImage(
            user=request.user,
            original_image=image_file,
            key=key,
            key_value=int(key_value),
            shuffle_indices=json.dumps(indices.tolist()),
            original_size=original_size_str,
        )
        encrypted_image_model.encrypted_image.save('encrypted_image.png', ContentFile(buffer.getvalue()), save=True)
        encrypted_image_model.save()

        print(f"[DEBUG] Total EncryptionPage() execution took {time.time() - full_start:.4f} seconds")

        return JsonResponse({
            'encrypted_image_url': encrypted_image_model.encrypted_image.url,
            'encryption_key': key,
            'system_generated': system_generated
        }, status=200)

    return render(request, 'Encryption.html')

@login_required
def VerifyOTP(request):
    if request.method == "POST":
        data = json.loads(request.body)
        image_id = data.get('image_id')
        entered_otp = data.get('otp')

        image_obj = get_object_or_404(EncryptedImage, id=image_id, user=request.user)

        if image_obj.otp_code == entered_otp:
            image_obj.otp_verified = True
            image_obj.save()
            return JsonResponse({'status': 'success', 'message': 'OTP verified successfully'})
        else:
            return JsonResponse({'status': 'fail', 'message': 'Invalid OTP'})

    return JsonResponse({'status': 'fail', 'message': 'Invalid request method'})

def send_otp_email(email_address, otp_code):
    subject = 'Your OTP Code'
    message = f'Your OTP code for verification is: {otp_code}'
    from_email = settings.DEFAULT_FROM_EMAIL
    recipient_list = [email_address]

    try:
        # Sending email
        response = send_mail(subject, message, from_email, recipient_list)
        print(f"Email sent, response: {response}")
        return True
    except Exception as e:
        print(f"[ERROR] in send_otp_email: {e}")
        return False

@login_required
@csrf_exempt
def SendOTPForImage(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            image_id = data.get("image_id")

            if not image_id:
                return JsonResponse({'status': 'fail', 'message': 'Image ID missing'}, status=400)

            image_obj = get_object_or_404(EncryptedImage, id=image_id, user=request.user)

            # Generate new OTP
            otp_code = str(random.randint(100000, 999999))
            image_obj.otp_code = otp_code
            image_obj.otp_verified = False
            image_obj.save()

            # Send OTP to user's email
            user_email = request.user.email
            if not user_email:
                return JsonResponse({'status': 'fail', 'message': 'Email missing. Please update your profile.'}, status=400)

            # Use the send_otp_email function
            if send_otp_email(user_email, otp_code):
                return JsonResponse({'status': 'success', 'message': 'OTP sent successfully to your email!'})
            else:
                return JsonResponse({'status': 'fail', 'message': 'Failed to send OTP email.'}, status=500)

        except Exception as e:
            print(f"[ERROR] in SendOTPForImage: {str(e)}")
            return JsonResponse({'status': 'fail', 'message': f'Something went wrong: {str(e)}'}, status=500)

    return JsonResponse({'status': 'fail', 'message': 'Invalid request method'}, status=405)

@login_required
def EncryptedImages(request):
    encrypted_images = EncryptedImage.objects.filter(user=request.user)

    for image in encrypted_images:
        if image.otp_verified:
            image.otp_verified = False
            image.save()

    return render(request, 'EncryptedImages.html', {
        'encrypted_images': encrypted_images
    })

@login_required
def SaveKey(request):
    data = json.loads(request.body.decode('utf-8'))
    key_value = data.get('key')

    if key_value:
        encrypted_image = EncryptedImage.objects.filter(user=request.user).last()
        if encrypted_image:
            encrypted_image.key = key_value
            encrypted_image.save()
            return JsonResponse({"message": "Key saved successfully"})
        else:
            return JsonResponse({"error": "No encrypted image found"}, status=400)
    else:
        return JsonResponse({"error": "Key value is missing"}, status=400)

def crop_to_original(padded_image, original_size):
    height, width, channels = original_size
    return padded_image[:height, :width, :channels]

def reverse_chaotic_sequence(image, iterations):
    N = image.shape[0]  # assume square block
    x, y = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    transformed_image = image.copy()
    for _ in range(iterations):
        new_x = (2 * x - y) % N
        new_y = (-x + y) % N
        transformed_image = transformed_image[new_x, new_y]
    return transformed_image

def reverse_shuffle_pixels(shuffled_image, indices):
    height, width, channels = shuffled_image.shape
    flat_pixels = shuffled_image.reshape(-1, channels)
    original_pixels = np.zeros_like(flat_pixels)
    for i, idx in enumerate(indices):
        original_pixels[idx] = flat_pixels[i]
    return original_pixels.reshape(height, width, channels)

def reverse_pixel_values(modified_image, key_value):
    height, width, channels = modified_image.shape
    modified_image = modified_image.astype(np.uint16)
    np.random.seed(key_value + 99)
    random_values = np.random.randint(1, 256, size=(height, width, channels), dtype=np.uint16)
    original_image = (modified_image - random_values + 256) % 256
    return original_image.astype(np.uint8)

def decrypt_image(encrypted_image, shuffle_indices, key_value, original_size):
    block_size = 64  # Same block size used in encryption

    # Step 1: Reverse pixel modification
    reversed_pixel_values = reverse_pixel_values(encrypted_image, key_value)

    # Step 2: Reverse pixel shuffling
    unshuffled_pixels = reverse_shuffle_pixels(reversed_pixel_values, shuffle_indices)

    # Step 3: Split into blocks and apply reverse Arnold Cat Map
    blocks, padded_shape = split_into_blocks(unshuffled_pixels, block_size)
    decrypted_blocks = []

    for (i, j), block in blocks:
        decrypted_block = reverse_chaotic_sequence(block, key_value)
        decrypted_blocks.append(((i, j), decrypted_block))

    # Step 4: Merge blocks
    decrypted_image = merge_blocks(decrypted_blocks, padded_shape)

    # Step 5: Crop back to original size
    cropped_image = crop_to_original(decrypted_image, original_size)
    return Image.fromarray(cropped_image.astype(np.uint8))

# ✅ 1. Decryption via Form Upload (check against all database)
@login_required
def DecryptionView(request):
    if request.method == "POST":
        uploaded_image = request.FILES.get('encrypted_image')
        entered_key = request.POST.get('key')

        if not uploaded_image and not entered_key:
            return JsonResponse({"error": "Please provide both encrypted image and key."})
        elif not uploaded_image:
            return JsonResponse({"error": "No image file selected."})
        elif not entered_key:
            return JsonResponse({"error": "Provide key that is used at encryption time."})

        uploaded_data = uploaded_image.read()
        uploaded_image.seek(0)

        for record in EncryptedImage.objects.all():
            with record.encrypted_image.open('rb') as stored_file:
                stored_data = stored_file.read()

                if uploaded_data == stored_data and entered_key == record.key:
                    try:
                        key_value = int(record.key_value)
                        shuffle_indices = json.loads(record.shuffle_indices)
                        original_size = json.loads(record.original_size)
                        image = Image.open(record.encrypted_image).convert('RGB')
                        image_array = np.array(image)

                        decrypted_image = decrypt_image(image_array, shuffle_indices, key_value, original_size)

                        decrypted_folder = os.path.join(settings.MEDIA_ROOT, "decrypted_images")
                        os.makedirs(decrypted_folder, exist_ok=True)
                        decrypted_filename = f"decrypted_{record.id}.png"
                        decrypted_path = os.path.join(decrypted_folder, decrypted_filename)
                        decrypted_image.save(decrypted_path)

                        # Construct the full URL for the decrypted image
                        decrypted_url = settings.MEDIA_URL + f"decrypted_images/{decrypted_filename}"

                        # Return the image URL in a JSON response
                        return JsonResponse({
                            "status": "success",
                            "decrypted_image_url": decrypted_url
                        })

                    except Exception as e:
                        return JsonResponse({
                            "status": "error",
                            "error": f"Error during decryption: {str(e)}"
                        })

        return JsonResponse({
            "status": "error",
            "error": "No matching encrypted image and key found!"
        })

    return render(request, "Decryption.html")

# ✅ 2. Decryption from EncryptedImages page (JSON based call)
@login_required
def DecryptionPage(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_id = data.get("image_id")
            key = data.get("secret_key")

            encrypted_image_obj = EncryptedImage.objects.filter(user=request.user, id=image_id, key=key).first()
            if not encrypted_image_obj:
                return JsonResponse({'error': 'No matching encrypted image found!'}, status=400)

            # ✅ Check OTP Verification
            if not encrypted_image_obj.otp_verified:
                return JsonResponse({'error': 'OTP verification required before decryption!'}, status=403)

            key_value = int(encrypted_image_obj.key_value)
            shuffle_indices = json.loads(encrypted_image_obj.shuffle_indices)
            original_size = json.loads(encrypted_image_obj.original_size)

            encrypted_image_path = encrypted_image_obj.encrypted_image.path
            image = Image.open(encrypted_image_path).convert('RGB')
            image_array = np.array(image, dtype=np.uint8)

            decrypted_image = decrypt_image(image_array, shuffle_indices, key_value, original_size)

            decrypted_folder = os.path.join(settings.MEDIA_ROOT, "decrypted_images")
            os.makedirs(decrypted_folder, exist_ok=True)
            decrypted_filename = f"decrypted_{image_id}.png"
            decrypted_path = os.path.join(decrypted_folder, decrypted_filename)
            decrypted_image.save(decrypted_path, format="PNG")

            decrypted_url = request.build_absolute_uri(settings.MEDIA_URL + f"decrypted_images/{decrypted_filename}")
            return JsonResponse({'status': 'success', 'decrypted_image_url': decrypted_url}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

# View to handle histogram request
@login_required
def HistogramPage(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            image_id = data.get("image_id")

            # 🔹 Retrieve image object
            image_obj = get_object_or_404(EncryptedImage, id=image_id)

            # 🔹 Load encrypted and original images in RGB mode
            encrypted_image = Image.open(image_obj.encrypted_image).convert('RGB')
            original_image = Image.open(image_obj.original_image).convert('RGB') if image_obj.original_image else None

            if not original_image:
                return JsonResponse({"status": "error", "error": "Original image not found in database"})

            # Convert to NumPy arrays efficiently
            encrypted_array = np.array(encrypted_image, dtype=np.uint8)
            original_array = np.array(original_image, dtype=np.uint8)

            # 🔹 Prepare figure
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=100)

            # 🔹 Encrypted Image Histogram
            axes[0].set_title("Encrypted Image Histogram")
            for i, color in enumerate(('red', 'green', 'blue')):
                axes[0].hist(encrypted_array[:, :, i].ravel(), bins=256, color=color, alpha=0.7, label=color)
            axes[0].legend()

            # 🔹 Original Image Histogram
            axes[1].set_title("Original Image Histogram")
            for i, color in enumerate(('red', 'green', 'blue')):
                axes[1].hist(original_array[:, :, i].ravel(), bins=256, color=color, alpha=0.7, label=color)
            axes[1].legend()

            # 🔹 Save histogram
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)

            histogram_path = os.path.join(settings.MEDIA_ROOT, "histogram.png")
            with open(histogram_path, "wb") as f:
                f.write(buf.getbuffer())

            histogram_url = request.build_absolute_uri(settings.MEDIA_URL + "histogram.png")

            return JsonResponse({"status": "success", "histogram_url": histogram_url})

        except Exception as e:
            return JsonResponse({"status": "error", "error": str(e)})
        
def generate_lyapunov_graph(image_path, base_key, output_path):
    if os.path.exists(output_path):
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    r_values = np.linspace(0.000001, 0.01, 100)

    # ✅ Load image only once
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image, dtype=np.uint8)

    plt.figure(figsize=(6, 4))  # ✅ Smaller figure size

    lyapunov_exponents = []

    # ✅ Pre-hash the base key once
    base_key_encoded = base_key.encode()

    for r in r_values:
        try:
            r_str = f"{r:.8f}"
            key2 = hashlib.sha256(base_key_encoded + r_str.encode()).hexdigest()

            image1, _, _, _ = encrypt_image(image_array, base_key)
            image2, _, _, _ = encrypt_image(image_array, key2)

            diff = np.array(image1, dtype=np.float64) - np.array(image2, dtype=np.float64)
            norm_diff = np.linalg.norm(diff) + 1e-10

            exponent = np.log(norm_diff)
            lyapunov_exponents.append(exponent)

        except Exception as e:
            print(f"Error at r = {r} for key {base_key}: {str(e)}")
            lyapunov_exponents.append(-100)

    lyapunov_exponents = np.array(lyapunov_exponents)
    plt.plot(r_values, lyapunov_exponents, linewidth=2, label=f'Key: {base_key}')

    plt.title("Lyapunov Exponent vs Perturbation")
    plt.xlabel("Perturbation in Key (r)")
    plt.ylabel("Lyapunov Exponent")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

@login_required
def LyapunovGraphPage(request):
    image_path = os.path.join(settings.BASE_DIR, 'myapp', 'static', 'Images', 'Flowers.jpg')
    output_path = os.path.join(settings.BASE_DIR, 'myapp', 'static', 'Images', 'lyapunov_plot.png')

    base_key = 'abc123'

    generate_lyapunov_graph(image_path, base_key, output_path)

    return render(request, 'Lyapunov.html', {
        'plot_url': 'Images/lyapunov_plot.png',
    })

# ✅ Encrypt one pixel only (mimic real encryption)
def encrypt_single_pixel(pixel_value, key):
    hash_val = hashlib.sha256(key.encode()).digest()
    modifier = hash_val[0]  # Get a single byte value (0–255)
    result = (int(pixel_value) + modifier) % 256
    return np.uint8(result)

# ✅ Optimized + Save-once Bifurcation graph
def generate_bifurcation_graph(image_path, base_key, output_path):
    if os.path.exists(output_path):
        return  # Already exists, no need to regenerate

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    r_values = np.linspace(0.000001, 0.01, 1000)
    pixel_values = []

    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image, dtype=np.uint8)

    # ✅ Pick one fixed pixel (R channel)
    original_pixel = image_array[100, 100, 0]
    base_key_encoded = base_key.encode()

    for r in r_values:
        try:
            r_str = f"{r:.8f}"
            modified_key = hashlib.sha256(base_key_encoded + r_str.encode()).hexdigest()

            encrypted_pixel = encrypt_single_pixel(original_pixel, modified_key)
            pixel_values.append(encrypted_pixel)

        except Exception as e:
            print(f"Error at r = {r}: {e}")
            pixel_values.append(0)

    # ✅ Save the plot
    plt.figure(figsize=(6, 4))
    plt.scatter(r_values, pixel_values, s=0.5, color='blue')
    plt.title("Bifurcation Graph")
    plt.xlabel("Key Perturbation (r)")
    plt.ylabel("Pixel Intensity at (100, 100)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


@login_required
def BifurcationPage(request):
    image_path = os.path.join(settings.BASE_DIR, 'myapp', 'static', 'Images', 'Flowers.jpg')
    output_path = os.path.join(settings.BASE_DIR, 'myapp', 'static', 'Images', 'bifurcation_plot.png')
    base_key = "abc123"

    generate_bifurcation_graph(image_path, base_key, output_path)

    return render(request, 'Bifurcation.html', {
        'plot_url': 'Images/bifurcation_plot.png'
    })

# Attack functions
def npcr(image1, image2):
    diff = np.not_equal(image1, image2)
    npcr_value = np.sum(diff) / diff.size * 100
    print("NPCR Value:", npcr_value)
    return round(npcr_value, 4)

def uaci(image1, image2):
    image1 = image1.astype(np.int16)
    image2 = image2.astype(np.int16)
    diff = np.abs(image1 - image2)
    uaci_value = np.sum(diff) / (255 * image1.size) * 100
    print("UACI Value:", uaci_value)
    return round(uaci_value, 4)

def baci(image1, image2):
    flat1 = image1.astype(np.uint8).flatten()
    flat2 = image2.astype(np.uint8).flatten()
    bits1 = np.unpackbits(flat1)
    bits2 = np.unpackbits(flat2)
    diff = np.abs(bits1 - bits2)
    baci_value = np.sum(diff) / bits1.size
    print("BACI Value:", baci_value)
    return round(baci_value, 4)

# Attacks Page
@login_required
def DifferentialAttacksPage(request, image_id):
    image_obj = get_object_or_404(EncryptedImage, id=image_id)
    original_image_path = image_obj.original_image.path
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    print("Original Image Loaded:", original_image is not None)

    if original_image is None:
        return JsonResponse({'error': 'Original image not found'}, status=404)

    original_image = cv2.resize(original_image, (256, 256))
    attack_result = None
    attacked_image_base64 = None
    attack_name = None

    if request.method == 'POST':
        print("POST received")

        if 'application/json' in request.content_type:
            data = json.loads(request.body)
            attack_name = data.get('attack_type')
            secret_key = data.get('secret_key')
        else:
            attack_name = request.POST.get('attack_type')
            secret_key = request.POST.get('secret_key')

        encrypted_original, _, _, _ = encrypt_image(original_image, secret_key)

        modified_key = secret_key + "1"
        modified_image = original_image.copy()
        x = random.randint(0, modified_image.shape[0] - 1)
        y = random.randint(0, modified_image.shape[1] - 1)
        channel = random.randint(0, 2)

        original_pixel_value = modified_image[x, y, channel]
        new_pixel_value = (int(original_pixel_value) + 1) % 256
        modified_image[x, y, channel] = new_pixel_value

        print(f"Pixel modified at ({x}, {y}, channel {channel})")
        print(f"Original pixel value: {original_pixel_value}")
        print(f"Modified pixel value: {new_pixel_value}")

        encrypted_modified, _, _, _ = encrypt_image(modified_image, modified_key)

        try:
            if attack_name == 'npcr':
                attack_result = npcr(encrypted_original, encrypted_modified)
            elif attack_name == 'uaci':
                attack_result = uaci(encrypted_original, encrypted_modified)
            elif attack_name == 'baci':
                attack_result = baci(encrypted_original, encrypted_modified)
        except Exception as e:
            print("❌ Error in attack calculation:", str(e))
            traceback.print_exc()
            attack_result = None

        if attack_result is not None:
            AttackResult.objects.create(
                image=image_obj,
                attack_type=attack_name,
                result_value=attack_result,
                modified_key_used=modified_key,
                pixel_coordinates=f"({x},{y},ch{channel})",
                original_pixel=int(original_pixel_value),
                modified_pixel=int(new_pixel_value)
            )

        attacked_pil_image = Image.fromarray(encrypted_modified)
        buffer = BytesIO()
        attacked_pil_image.save(buffer, format='PNG')
        attacked_image_base64 = base64.b64encode(buffer.getvalue()).decode()

        return JsonResponse({
            'attacked_image_url': attacked_image_base64,
            'attack_result': attack_result,
            'attack_name': attack_name
        })

    original_pil_image = Image.fromarray(original_image)
    buffer = BytesIO()
    original_pil_image.save(buffer, format='PNG')
    original_image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return render(request, 'DifferentialAttacks.html', {
        'image': image_obj,
        'original_image': original_image_base64,
    })
