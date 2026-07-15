const imageInput = document.getElementById("image");

if (imageInput) {

    imageInput.addEventListener("change", function () {

        const file = this.files[0];
        const fileName = document.getElementById("file-name");

if (fileName) {
    fileName.textContent = file.name;
}

        if (!file) return;

        // Check image type
        if (!file.type.startsWith("image/")) {
            alert("Please select a valid image file.");
            this.value = "";
            return;
        }

        // Preview Image
        const reader = new FileReader();

        reader.onload = function (e) {

            let preview = document.getElementById("preview");

            if (!preview) {

                preview = document.createElement("img");

                preview.id = "preview";

                preview.style.width = "250px";
                preview.style.marginTop = "20px";
                preview.style.borderRadius = "10px";
                preview.style.display = "block";
                preview.style.marginLeft = "auto";
                preview.style.marginRight = "auto";

                document.querySelector(".upload-box").appendChild(preview);
            }

            preview.src = e.target.result;
        };

        reader.readAsDataURL(file);

    });

}