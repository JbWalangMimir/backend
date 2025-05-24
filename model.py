from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

class EfficientNetModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.img_size = (224, 224)  # Adjust based on your model's expected input
    
    def preprocess_image(self, img_bytes):
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize if your model expects this
        return img_array
    
    def predict_top5(self, img_bytes):
        processed_img = self.preprocess_image(img_bytes)
        predictions = self.model.predict(processed_img)
        # Assuming your model outputs probabilities for multiple classes
        top5_indices = np.argsort(predictions[0])[-5:][::-1]
        top5_probs = predictions[0][top5_indices]
        
        # Replace with your actual class labels
        class_labels = ['Acanthophippium sylhetense', 'Aerides leeana', 'Aerides magnifica', 'Amesiella monticola', 'Aphyllorchis montana', 'Arundina graminifolia', 'Bulbophyllum basisetum', 'Bulbophyllum catenulatum', 'Bulbophyllum ocellatum', 'Bulbophyllum romyi', 'Calanthe furcata', 'Calanthe lyroglossa', 'Calanthe rubens', 'Ceratostylis incognita', 'Coelogyne pandurata', 'Cymbidium aloifolium', 'Dendrobium amethystoglossum', 'Dendrobium anosmum', 'Dendrobium chameleon', 'Dendrobium goldschmidtianum', 'Dendrobium macrophyllum', 'Dendrobium parthenium', 'Dendrobium sanderae', 'Dendrobium uniflorum', 'Dendrobium victoriae-reginae', 'Dendrochilum cootesii', 'Dendrochilum glumaceum', 'Dendrochilum tenellum', 'Grammatophyllum multiflorum', 'Liparis bootanensis', 'Liparis nervosa', 'Ludisia discolor', 'Malaxis monophyllos', 'Microtis unifolia', 'Paphiopedilum haynaldianum', 'Phalaenopsis amabilis', 'Phalaenopsis aphrodite', 'Phalaenopsis cornucervi', 'Phalaenopsis deliciosa', 'Phalaenopsis equestris', 'Phalaenopsis hieroglyphica', 'Phalaenopsis lindenii', 'Phalaenopsis lueddemanniana', 'Phalaenopsis philippinensis', 'Phalaenopsis schilleriana', 'Phalaenopsis stuartiana', 'Phalaenopsis sumatrana', 'Plocoglottis plicata', 'Rhynchostylis gigantea', 'Rhynchostylis retusa', 'Robiquetia cerina', 'Robiquetia ilocosnortensis', 'Spathoglottis plicata', 'Spiranthes australis', 'Spiranthes sinensis', 'Trachoma minuta', 'Trichoglottis atropurpurea', 'Trichoglottis bataanensis', 'Vanda barnesii', 'Vanda lamellata', 'Vanda sanderiana', 'Zeuxine nervosa', 'Zeuxine strateumatica']  
        
        return [
            {"class": class_labels[i], "probability": float(prob)}
            for i, prob in zip(top5_indices, top5_probs)
        ]