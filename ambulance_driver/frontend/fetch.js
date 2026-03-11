async function uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('http://YOUR_IP_ADDRESS:8000/analyze', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    if (result.success) {
        alert(`Severity: ${result.severity_score}\nHospital Needed: ${result.hospital_tier}`);
    }
}