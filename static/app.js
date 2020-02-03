

const handleImageUpload = event => {
    const files = event.target.files
    const formData = new FormData()
    formData.append('file', files[0])
  
    fetch('/http://127.0.0.1:5000/api/file-upload', {
      method: 'POST',
      body: formData,
      headers: { 
        "Content-type": "multipart/form-data;"
        }
    })
    .then(response => response.json())
    .then(data => {
      console.log(data)
    })
    .catch(error => {
      console.error(error)
    })
  }

document.querySelector('#fileUpload').addEventListener('change', event => {
    handleImageUpload(event)
  })