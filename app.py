from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)

# Warehouse coordinates (Isheri Osun)
WAREHOUSE_LAT = 6.5244
WAREHOUSE_LNG = 3.3103

DELIVERY_DAYS = 4
ORDERS_PER_BATCH = 25


def create_maps_url(batch_df):
    origin = f"{WAREHOUSE_LAT},{WAREHOUSE_LNG}"

    waypoints = [
        f"{row['latitude']},{row['longitude']}"
        for _, row in batch_df.iterrows()
    ]

    waypoints_str = "|".join(waypoints)

    return (
        f"https://www.google.com/maps/dir/?api=1"
        f"&origin={origin}"
        f"&destination={origin}"
        f"&travelmode=driving"
        f"&waypoints={waypoints_str}"
    )


@app.route("/plan-deliveries", methods=["POST"])
def plan_deliveries():
    data = request.json
    orders = data["orders"]

    df = pd.DataFrame(orders)

    total_orders = len(df)
    num_batches = max(1, total_orders // ORDERS_PER_BATCH)

    coords = df[['latitude', 'longitude']]

    kmeans = KMeans(n_clusters=num_batches, random_state=42)
    df['batch'] = kmeans.fit_predict(coords)

    df['Day'] = (df['batch'] % DELIVERY_DAYS) + 1

    def distance(row):
        return np.sqrt(
            (row['latitude'] - WAREHOUSE_LAT)**2 +
            (row['longitude'] - WAREHOUSE_LNG)**2
        )

    df['dist'] = df.apply(distance, axis=1)
    df = df.sort_values(['batch', 'dist']).reset_index(drop=True)

    df['Batch'] = df['batch'] + 1

    batch_urls = {}
    for batch in df['Batch'].unique():
        batch_df = df[df['Batch'] == batch]
        batch_urls[batch] = create_maps_url(batch_df)

    df['route_url'] = df['Batch'].apply(lambda x: batch_urls[x])

    results = df[['order_id', 'Day', 'Batch', 'route_url']].to_dict(orient="records")

    drivers_needed = df['Batch'].nunique()

    return jsonify({
        "drivers_needed": drivers_needed,
        "results": results
    })


@app.route("/")
def home():
    return "Logistics API running."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
