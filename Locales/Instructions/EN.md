# How to use

## 1. Load Model and Set

Go to "Settings", the first part is model settings.

- GPU Layers: Input -1 to unload all layers to GPU, input 0 to use pure CPU, or input a value bigger than 0 to use CPU and GPU mixed inference.
- Select Model: The program will detect `gguf` files in the `models` folder, and list them here.
- Model Version: Usually auto-updated, no need to manage.

The second part is generation settings.
We'll focus on the first two and last two settings since users who can modify the rest two settings are probably not needed to read this tutorial.

- Context Length: This value determines how long the model can handle HTML files, and the following section will explain how to determine whether the input is too long. But, the higher this value is, the more VRAM/RAM it will consume. Changing this value requires reloading the model to take effect.
- Max New Tokens: Determines how many tokens the model can say, changing this value doesn't require reloading the model.
- Remove Outer Code Block: The usual habit of V2 models, it will wrap the conversion result in code blocks. In copying, whether to remove the code blocks is decided by you. In rendering, it will always remove the code blocks.

The third part is HTML settings.
We'll focus on the first setting.

- Clean HTML: Pre-clean up some things that are likely to be useless(such as CSS and scripts), shortening the HTML length.

## 2. Start Converting

Go to "Generation", you can see two columns.
Firstly, from the left column.

The left column is mainly input and preview HTML, you can use two types of input, URL or HTML file. The former's priority is higher, so if you have both URL and file, the file will be ignored.
After entering the URL, you need to submit to start getting HTML. And entering the file will read the HTML immediately.
Got HTML, the HTML preview will be rendered below the file input box, this preview is usually safe, it won't crash the page, unlike the HTML tab's rendering function.
Then, if the HTML content is updated, the program will automatically calculate the Token number, if the number is too large, there will be a corresponding warning. Although it will give you an estimate of the predicted number of cleaned HTML, it's not accurate, depending on the cleaning settings, the result may vary.

The right column is pretty simple, mainly the generate button, copy button and stop button, you'll get it at the first glance.
Note: if the model is in the prefilling stage (not generating), clicking the stop button will not stop prefilling.
