# Training Guide Using MLX (Recommended for Mac)

## System Requirements
- Mac OS 14 or later

## Steps
1. **Download All Files from LLAMA_FORMAT_MLX**
   Download all files to `model_path` (the path can be customized).

2. **Prepare Data**

   ### Method 1: Use Demo Data
   - Run `MiniCPM/finetune/data_processing.ipynb` for data processing.
   - Obtain `train.json` and `dev.json` files located under `MiniCPM/finetune/data/mlx_AdvertiseGen`.

   ### Method 2: Custom Data
   - Place the training and testing data files `train.json` and `dev.json` in your dataset directory:
     ```
     -data_path
         -train.json
         -dev.json
     ```
   - Ensure the JSON data follows this format: each line contains a dictionary, each dictionary represents a single data entry, and the dictionary has three keys: `prompt`, `input`, and `output`:
     ```json
     {"input": "type#dress*material#lace*style#royal*pattern#embroidery*pattern#lace*dress_shape#full_skirt*dress_hem#lace*dress_sleeve#puff_sleeve", "prompt": "\nPlease generate an advertisement slogan for the following keywords.", "output": "The royal style of sweet lace design, with clear lace seams, custom embroidered shell edging, adds a touch of refinement to the dress. The extra-wide skirt, paired with delicate lace trim, exudes a fairy-like allure and femininity when worn. The raised floral fabric on the puff sleeves adds a romantic feel to the details, making you the fairy princess of spring. Romantic lace adorns the entire dress body, bright and radiant, exuding an ethereal charm."}
     {"input": "type#pants*fit#slim*color#black*style#minimalist*pants_length#ankle", "prompt": "\nPlease generate an advertisement slogan for the following keywords.", "output": "The personalized ankle-length pant style, when worn, visually elongates your body proportions, making you look more stylish. The minimalist black series exudes a fashionable charm, fully highlighting your unique mature allure. The sculpted three-dimensional silhouette shapes a slender curve."}
     {"input": "type#dress*fit#slim*style#literary*style#minimalist*pattern#print*pattern#contrast*dress_hem#pleated*dress_length#dress*dress_neckline#round", "prompt": "\nPlease generate an advertisement slogan for the following keywords.", "output": "The literary and individualistic printed dress, with a navy blue base, is low-key yet grand. Contrast sunflowers spread across the entire dress body, vibrant and beautiful, bringing a youthful and fashionable aura. The basic comfortable round neckline is simple yet elegant, outlining a refined face. A button closure at the back makes it easy to put on and take off. The front panel features a three-dimensional pleated design, complemented by the back panel's pleating craftsmanship, adding layers and spatial sense, slimming and shaping."}
     ```

3. **Start Training After Data Preparation**
   ```sh
   cd MiniCPM/finetune
   python mlx_finetune.py --model model_path --data data/mlx_AdvertiseGen --train --seed 2024 --iters 500
   ```

4. **Find the `adapters.npz` File and Start Testing**
   ```sh
   python mlx_finetune.py --model model_path --data data/mlx_AdvertiseGen --test --seed 2024 --adapter-file adapters.npz
   ```

5. **Perform Inference Using the Following Command**
   ```sh
   python mlx_finetune.py --model model_path --seed 2024 --prompt '<User>#zip*dress_style#strapless*dress_style#waist_cinched\nPlease generate an advertisement slogan for the above keywords.<AI>' --adapter-file adapters.npz
   ```
