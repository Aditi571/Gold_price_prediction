document.getElementById("dataForm").addEventListener("submit", function(event) {
    event.preventDefault();

    const formData = new FormData(this);
    fetch("/submit", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => alert("Data submitted successfully!"))
    .catch(error => alert("Error submitting data: " + error));
});
