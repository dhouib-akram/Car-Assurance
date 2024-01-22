import streamlit as st
import requests

car_brand_ctaegory = {
    'Aston Martin': 'Luxury',
    'Mercedes-Benz': 'Luxury',
    'Mini': 'Standard',
    'Tesla': 'Electric',
    'GMC': 'SUV',
    'Alfa Romeo': 'Sport',
    'Studebaker': 'Classic',
    'Suzuki': 'Standard',
    'Peugeot': 'Standard',
    'Genesis': 'Luxury',
    'BMW': 'Luxury',
    'Honda': 'Standard',
    'Chrysler': 'Standard',
    'Mazda': 'Standard',
    'Infiniti': 'Luxury',
    'Land Rover': 'SUV',
    'Dodge': 'Standard',
    'Fiat': 'Standard',
    'Maserati': 'Luxury',
    'Saab': 'Standard',
    'Nissan': 'Standard',
    'Hudson': 'Classic',
    'Lincoln': 'Luxury',
    'Volvo': 'Luxury',
    'Mitsubishi': 'Standard',
    'Oldsmobile': 'Classic',
    'Lexus': 'Luxury',
    'Buick': 'Luxury',
    'Jaguar': 'Luxury',
    'Toyota': 'Standard',
    'Volkswagen': 'Standard',
    'Renault': 'Standard',
    'Citroen': 'Standard',
    'Audi': 'Luxury',
    'Subaru': 'Standard',
    'Cadillac': 'Luxury',
    'Pontiac': 'Standard',
    'Porsche': 'Sport',
    'Daewoo': 'Standard',
    'Bugatti': 'Exotic',
    'Jeep': 'SUV',
    'Ram Trucks': 'Truck',
    'Chevrolet': 'Standard',
    'MG': 'Sport',
    'Hyundai': 'Standard',
    'Ferrari': 'Exotic',
    'Acura': 'Luxury',
    'Kia': 'Standard',
    'Bentley': 'Luxury',
    'Ford': 'Standard',
}

cat_repair_cost = {
    'Luxury': {'Dent': {'Minor': 500, 'Moderate': 1000, 'Severe': 2000},
               'Scratch': {'Minor': 300, 'Moderate': 700, 'Severe': 1500},
               'Crack': {'Minor': 800, 'Moderate': 1200, 'Severe': 2500},
               'Glass Shatter': {'Minor': 1000, 'Moderate': 1800, 'Severe': 3000},
               'Lamp Broken': {'Minor': 600, 'Moderate': 1000, 'Severe': 2000},
               'Tire Flat': {'Minor': 200, 'Moderate': 400, 'Severe': 800}},
    'Standard': {'Dent': {'Minor': 400, 'Moderate': 850, 'Severe': 1600},
                 'Scratch': {'Minor': 200, 'Moderate': 550, 'Severe': 1200},
                 'Crack': {'Minor': 700, 'Moderate': 1050, 'Severe': 2200},
                 'Glass Shatter': {'Minor': 800, 'Moderate': 1500, 'Severe': 2700},
                 'Lamp Broken': {'Minor': 500, 'Moderate': 900, 'Severe': 1800},
                 'Tire Flat': {'Minor': 100, 'Moderate': 200, 'Severe': 400}},
    'Sport': {'Dent': {'Minor': 600, 'Moderate': 1100, 'Severe': 2100},
              'Scratch': {'Minor': 350, 'Moderate': 800, 'Severe': 1600},
              'Crack': {'Minor': 900, 'Moderate': 1300, 'Severe': 2600},
              'Glass Shatter': {'Minor': 1100, 'Moderate': 2000, 'Severe': 3200},
              'Lamp Broken': {'Minor': 700, 'Moderate': 1200, 'Severe': 2300},
              'Tire Flat': {'Minor': 150, 'Moderate': 300, 'Severe': 600}},
    'Electric': {'Dent': {'Minor': 700, 'Moderate': 1200, 'Severe': 2300},
                 'Scratch': {'Minor': 400, 'Moderate': 900, 'Severe': 1800},
                 'Crack': {'Minor': 1000, 'Moderate': 1400, 'Severe': 2700},
                 'Glass Shatter': {'Minor': 1200, 'Moderate': 2200, 'Severe': 3400},
                 'Lamp Broken': {'Minor': 800, 'Moderate': 1300, 'Severe': 2400},
                 'Tire Flat': {'Minor': 180, 'Moderate': 360, 'Severe': 720}},
    'SUV': {'Dent': {'Minor': 500, 'Moderate': 950, 'Severe': 1800},
            'Scratch': {'Minor': 250, 'Moderate': 600, 'Severe': 1300},
            'Crack': {'Minor': 800, 'Moderate': 1100, 'Severe': 2200},
            'Glass Shatter': {'Minor': 900, 'Moderate': 1600, 'Severe': 2800},
            'Lamp Broken': {'Minor': 550, 'Moderate': 1000, 'Severe': 2000},
            'Tire Flat': {'Minor': 120, 'Moderate': 240, 'Severe': 480}},
    'Classic': {'Dent': {'Minor': 300, 'Moderate': 700, 'Severe': 1500},
                'Scratch': {'Minor': 150, 'Moderate': 500, 'Severe': 1100},
                'Crack': {'Minor': 600, 'Moderate': 1000, 'Severe': 2000},
                'Glass Shatter': {'Minor': 700, 'Moderate': 1300, 'Severe': 2500},
                'Lamp Broken': {'Minor': 400, 'Moderate': 800, 'Severe': 1700},
                'Tire Flat': {'Minor': 200, 'Moderate': 400, 'Severe': 800}},
}

API_URL = "https://api-inference.huggingface.co/models/dima806/car_brand_image_detection"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def query(uploaded_file):
    try:
        # Use BytesIO to read the file content as bytes
        file_content = uploaded_file.read()
        response = requests.post(API_URL, headers=headers, data=file_content)
        return response.json()
    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    st.title("Car Repair Cost Estimation App")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image of your car...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("Estimate Repair Cost"):
            # Perform car brand detection
            output = query(uploaded_file)
            car_brand = output[0]['label']
            
            # Get repair cost category
            repair_cost_category = car_brand_ctaegory.get(car_brand, 'Unknown')
            
            # Display car brand and repair cost category
            st.subheader("Car Brand Detection Result:")
            st.text(f"Detected Car Brand: {car_brand}")
            st.text(f"Repair Cost Category: {repair_cost_category}")
            
            # Display repair cost details
            st.subheader("Repair Cost Details:")
            if repair_cost_category != 'Unknown':
                repair_costs = cat_repair_cost.get(repair_cost_category, {})
                for damage_type, damage_costs in repair_costs.items():
                    st.write(f"**{damage_type}**:")
                    for severity, cost in damage_costs.items():
                        st.write(f"- {severity.capitalize()}: ${cost}")
            else:
                st.warning("Repair cost category is unknown for the detected car brand.")
            
if __name__ == "__main__":
    main()