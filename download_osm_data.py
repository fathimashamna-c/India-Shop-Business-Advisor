@st.cache_data(ttl=21600, show_spinner=False)
def live_osm_places(lat, lon, radius_m):
    """
    Fetch nearby mapped places from Overpass.
    Tries two endpoints, with short timeouts.
    Cache successful data for 6 hours.
    """

    query = f"""
    [out:json][timeout:8];
    (
      nwr(around:{int(radius_m)},{lat},{lon})["shop"];
      nwr(around:{int(radius_m)},{lat},{lon})
        ["amenity"~"^(cafe|restaurant|fast_food|food_court|school|hospital|clinic|pharmacy|college|university|kindergarten|doctors|dentist)$"];
      nwr(around:{int(radius_m)},{lat},{lon})
        ["healthcare"~"^(hospital|clinic|pharmacy|doctor|dentist)$"];
      nwr(around:{int(radius_m)},{lat},{lon})
        ["craft"~"^(electronics_repair|shoemaker|tailor|laundry)$"];
    );
    out center tags;
    """

    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    errors = []

    for endpoint in endpoints:
        try:
            response = requests.post(
                endpoint,
                data={"data": query},
                headers=HEADERS,
                timeout=(1.5, 5),
            )

            response.raise_for_status()
            payload = response.json()

            elements = payload.get("elements", [])

            places = []

            for item in elements:
                tags = item.get("tags", {})

                places.append({
                    "name": tags.get("name", ""),
                    "shop": tags.get("shop", ""),
                    "amenity": tags.get("amenity", ""),
                    "craft": tags.get("craft", ""),
                    "healthcare": tags.get("healthcare", ""),
                    "category": "",
                })

            return places, None

        except requests.exceptions.Timeout:
            errors.append("Server timeout")

        except requests.exceptions.RequestException as exc:
            errors.append(f"Network error: {str(exc)[:150]}")

        except (ValueError, TypeError):
            errors.append("Invalid server response")

    return None, (
        "Live OSM servers unavailable. "
        + " | ".join(errors)
    )