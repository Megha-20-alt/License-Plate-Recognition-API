
async function predictPlate() {

    const fileInput =
        document.getElementById("imageInput");

    const result =
        document.getElementById("result");

    const loading =
        document.getElementById("loading");

    const preview =
        document.getElementById("preview");


    result.innerHTML = "";


    if (fileInput.files.length === 0) {

        result.innerHTML =
            '<p class="error">Please select an image.</p>';

        return;
    }


    const file = fileInput.files[0];


    // Show image preview

    preview.innerHTML =
        `<img src="${URL.createObjectURL(file)}">`;


    // Create form data

    const formData = new FormData();

    formData.append("file", file);


    loading.style.display = "block";


    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        loading.style.display = "none";


        if (!response.ok) {

            throw new Error(
                "Prediction failed"
            );
        }


        // No plates detected

        if (data.plates.length === 0) {

            result.innerHTML =
                "<p>No license plate detected.</p>";

            return;
        }


        result.innerHTML =
            "<h3>Detection Result</h3>";


        // Display detected plates

        data.plates.forEach(
            (plate, index) => {

                result.innerHTML += `

                    <div class="plate">

                        <strong>
                            Plate ${index + 1}
                        </strong>

                        <p>
                            Number:
                            <strong>
                                ${plate.text || "Not recognized"}
                            </strong>
                        </p>

                        <p>
                            Confidence:
                            ${(plate.confidence * 100).toFixed(2)}%
                        </p>

                    </div>

                `;

            }
        );


    } catch (error) {

        loading.style.display = "none";


        result.innerHTML = `
            <p class="error">
                Could not process the image.
            </p>
        `;


        console.error(error);
    }
}

