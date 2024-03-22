import os
from django.http import JsonResponse
from transformers import UdopProcessor, UdopForConditionalGeneration
from django.views.decorators.csrf import csrf_exempt

repo_id = "microsoft/udop-large"

processor = UdopProcessor.from_pretrained(repo_id)
model = UdopForConditionalGeneration.from_pretrained(repo_id)

from PIL import Image


@csrf_exempt
def inference_view(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt')
        image = request.FILES.get('image')

        # Save the uploaded image temporarily
        image_path = 'temp_image.png'
        with open(image_path, 'wb') as f:
            f.write(image.read())

        # Perform inference
        result = perform_inference(prompt, image_path)

        # Delete the temporary image file
        os.remove(image_path)

        return JsonResponse({'result': result})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)


@csrf_exempt
def perform_inference(prompt, image_path, max_tokens=20):
    image = Image.open(image_path).convert("RGB")
    encoding = processor(images=image, text=prompt, return_tensors="pt")
    outputs = model.generate(**encoding, max_new_tokens=max_tokens)
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return generated_text
