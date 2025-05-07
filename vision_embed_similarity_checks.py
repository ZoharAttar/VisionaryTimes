import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import os
from PIL import Image
import clip
from torchvision import models, transforms
from transformers import AutoProcessor, AutoModel
import torchvision.transforms as T
from transformers import SiglipModel, SiglipProcessor, SiglipVisionModel
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Base directory for plots
base_save_dir = '/home/arielsi/VisionaryTimes/synthetic_plots'

# ===========================
# DATA GENERATION FUNCTIONS
# ===========================

def generate_gradual_increase(size=500):
    return np.linspace(0, 10, size) + np.random.normal(0, 0.5, size)

def generate_sharp_changes(size=500):
    noise = np.random.normal(0, 1, size)
    shift_point = size // 2
    y = np.zeros(size)
    y[:shift_point] = noise[:shift_point] - 1.5
    y[shift_point:] = noise[shift_point:] + 1.0
    return y

def generate_noise(size=500):
    return np.random.normal(0, 3, size)

def create_plot(data, plot_title, save_path):
    plt.figure(figsize=(5, 5))
    plt.plot(data)
    plt.title(plot_title)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

# ===========================
# PLOT GENERATION ONLY
# ===========================

def generate_and_save_plots():
    os.makedirs(base_save_dir, exist_ok=True)
    for category, gen_func in zip(['Gradual_Increase', 'Sharp_Changes', 'Noise'],
                                  [generate_gradual_increase, generate_sharp_changes, generate_noise]):
        for i in range(20):
            data = gen_func()
            plot_title = f"{category}_Plot_{i+1}"
            plot_path = os.path.join(base_save_dir, f"{category}_{i+1}.png")
            create_plot(data, plot_title, plot_path)

# ===========================
# EMBEDDING FUNCTIONS
# ===========================

def load_encoder(encoder_name):
    if encoder_name == "CLIP": #vision-language transformer trained to match images and text in a shared embedding space (512-dimensional space).
        model, preprocess = clip.load("ViT-B/32", device=device) #A Vision Transformer (base size) with a patch size of 32×32 pixels.
        model.eval() # Puts the model in inference mode (disables dropout).
        return model, preprocess 

    elif encoder_name == "ViT": #ViT applies a transformer architecture by splitting images into patches and processing them like sequences.
        vit_model = models.vit_b_16(pretrained=True) # Vision Transformer with base size and 16×16 input patches. pretrained=True: Loads weights pretrained on ImageNet.
        vit_model.heads = torch.nn.Identity() #Removes the classification head to get raw image features.
        vit_model.to(device).eval()
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]) #Standard preprocessing to match ImageNet-trained ViT input format (224×224, normalized RGB).
        return vit_model, preprocess

    elif encoder_name == "ResNet": #a CNN that uses skip connections to train, helping to prevent vanishing gradients.
        resnet = models.resnet50(pretrained=True)
        resnet.fc = torch.nn.Identity()
        resnet.to(device).eval()
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return resnet, preprocess

    elif encoder_name == "DINOv2": #DINOv2 is a self-supervised vision transformer model that learns high-quality image representations without labeled data.
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device).eval()
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return dinov2, preprocess

    elif encoder_name == "SigLIP": #SigLIP (Sigmoid Loss for Language-Image Pretraining) is a vision-language model that replaces CLIP’s contrastive loss with a sigmoid-based loss.
        processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224") # Loads the image processor (e.g., tokenizer and image transformer)
        model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224") # Loads the pre-trained SigLIP vision encoder
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) #  by using mean=0.5 and std=0.5, it maps [0, 1] → [-1, 1], which matches the normalization used during SigLIP's pretraining.
        ])
        return model, preprocess
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")

def generate_embeddings_from_saved_plots(encoder_name):
    encoder, preprocess = load_encoder(encoder_name)
    encoder.to(device)  # Ensure the model is moved to the same device as the input
    embeddings = []
    labels = []
    for category in ['Gradual_Increase', 'Sharp_Changes', 'Noise']:
        for i in range(1, 21):
            img_path = os.path.join(base_save_dir, f"{category}_{i}.png")
            image = Image.open(img_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)  # Move the input to the correct device

            with torch.no_grad():
                if encoder_name == "CLIP":
                    features = encoder.encode_image(image_input)
                elif encoder_name == "SigLIP":
                    outputs = encoder(image_input)
                    features = outputs.last_hidden_state.mean(dim=1)
                else:
                    features = encoder(image_input)

            embeddings.append(features.cpu().numpy())
            labels.append(category)

    embeddings = np.array(embeddings).reshape(len(embeddings), -1)
    return embeddings, labels

# ===========================
# VISUALIZATION
# ===========================

def plot_similarity_matrix(embeddings, encoder_name):
    sim_matrix = cosine_similarity(embeddings)
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap='viridis')
    plt.title(f"Cosine Similarity Heatmap - {encoder_name}")
    out_path = os.path.join(base_save_dir, f"{encoder_name}_embedding_similarity_heatmap.png")
    plt.savefig(out_path)
    plt.show()


def tsne_visualization(embeddings, labels, encoder_name):
    # t-SNE reduce high-dim embeddings to 2D for visualization in order to keep similar points close together and dissimilar points farther apart.
    # n_components=2 → project into 2 dimensions.
    # perplexity=10 → balances attention between local and global structure (good for ~60 samples)
    tsne = TSNE(n_components=2, random_state=42, perplexity=10) #Low perplexity (e.g. 5–30) focuses on local neighborhoods, High perplexity (e.g. 30–50+) focuses on global structure.
    reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette='deep')
    plt.title(f"t-SNE - {encoder_name}")
    plt.legend()
    plt.savefig(os.path.join(base_save_dir, f"{encoder_name}_tsne.png"))
    plt.show()


def evaluate_clustering(embeddings, labels, encoder_name):
    # Silhouette Score is a way to evaluate how well your data has been clustered, 
    # combining measures of cohesion (how close points are within the same cluster) 
    # and separation (how far points are from other clusters).
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, cluster_labels) # The Silhouette Score is a way to evaluate how well your data has been clustered, 
                                                         # combining measures of cohesion (how close points are within the same cluster) 
                                                         # and separation (how far points are from other clusters).
    
    # Plot using first two dimensions of raw embeddings
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1],
                    hue=cluster_labels, palette='Set2', legend="full")
    plt.title(f"KMeans Clustering - {encoder_name}\nSilhouette Score: {score:.3f}")
    plt.xlabel("Embedding Dim 0")
    plt.ylabel("Embedding Dim 1")
    plt.legend(title="Cluster")    
    plt.savefig(os.path.join(base_save_dir, f"{encoder_name}_clustering.png"))
    plt.show()

# ===========================
# MAIN USAGE
# ===========================

# Step 1: Generate plots if not already done
# generate_and_save_plots()

# Step 2: For each encoder, generate embeddings and plot similarity
for encoder_name in ['CLIP', 'ViT', 'ResNet', 'DINOv2', 'SigLIP']:
    print(f"Processing encoder: {encoder_name}")
    embeddings, labels = generate_embeddings_from_saved_plots(encoder_name)
    
    # np.save(os.path.join(base_save_dir, f"{encoder_name}_synthetic_embeddings.npy"), embeddings)
    
    # plot_similarity_matrix(embeddings, encoder_name)
    
    # tsne_visualization(embeddings, labels, encoder_name)
    
    evaluate_clustering(embeddings, labels, encoder_name)

print("All embeddings generated and plots saved.")
