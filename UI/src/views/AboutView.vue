<template>
  <div class="about">
    <div class="about" v-if="!loading">
      <input v-if="!show_result" type="file" @change="handleFileChange" class="file-input" />
      <button v-if="!show_result" @click="analyzeFile" class="analyze-button">Analyze</button>
    </div>
    <div v-if="loading" class="loading">
      <trinity-rings-spinner
        :animation-duration="1500"
        :size="100"
        color="#00CC99"
      />
    </div>
    <div class="image-container" v-if="show_result">
      <div class="row">
        <div class="column">
          <img v-if="preview" :src="preview" style="max-width: 500px; max-height: 500px;">
        </div>
        <div class="column">
          <img :src="imageSrc" style="max-width: 500px; max-height: 500px;margin-left: 50px;" alt="Detection result" class="image-with-text"/>
        </div>
      </div>
      <!--<div class="overlay-square"></div>-->
      <!--<p class="text-below"><h3>Detection result:</h3><p class="red">BENIGN</p></p>-->
      <button @click="resetAnalyze" class="back-button">Back</button>
    </div>
 </div>
</template>

<style>
@media (min-width: 1024px) {
  .about {
    min-height: 100vh;
    display: flex;
    align-items: center;
  }
  img.image-with-text {
  max-width: 100%;
  height: auto;
  display: block;
}
.file-input {
  background-color: #f2f2f2;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  margin: 5px;
}
.analyze-button {
  background-color: hsla(160, 100%, 37%, 1);
  color: #fff;
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
.back-button {
  background-color: hsla(160, 100%, 37%, 1);
  color: #fff;
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin-left: 168px;
  margin-top: 10px;
}
.image-container {
  position: relative;
  width: 300px; /* Ajustați dimensiunile containerului în funcție de nevoile dvs. */
}
.red {
  color: red;
  font-weight: bold;
}
img {
  max-width: 100%;
  height: auto;
  display: block;
}
.overlay-square {
  position: absolute;
  top: 170px; /* Ajustați poziția pătratului în funcție de nevoile dvs. */
  left: 155px; /* Ajustați poziția pătratului în funcție de nevoile dvs. */
  width: 80px; /* Ajustați dimensiunile pătratului în funcție de nevoile dvs. */
  height: 120px; /* Ajustați dimensiunile pătratului în funcție de nevoile dvs. */
  border: 2px solid red; /* Adăugați un chenar în jurul pătratului */
  background-color: transparent;
  /* Alte stiluri de personalizare a pătratului */
}
.text-below {
  text-align: center;
  margin-top: 10px; /* Ajustați spațiul între pătrat și textul de sub el */
  /* Alte stiluri de personalizare pentru textul de sub pătrat */
}
.image-with-text::before {
  content: "hehe"; /* Textul pe imagine */
  position: absolute;
  top: 100;
  left: 120;
  background-color: rgba(255, 0, 0, 0.5); /* Fundal pentru text */
  color: white; /* Culoarea textului */
  padding: 5px; /* Spațiu în jurul textului */
  /* Alte stiluri de personalizare pentru text */
}
h3 {
  font-size: 1.2rem;
  font-weight: 500;
  margin-bottom: 0.4rem;
  color: var(--color-heading);
}
.column {
  float: left;
  width: 50%;
  padding: 5px;
}

/* Clear floats after image containers */
.row::after {
  content: "";
  clear: both;
  display: table;
}
}
</style>

<script>
import { TrinityRingsSpinner } from 'epic-spinners'
import axios from 'axios';
export default {
  name: 'AboutView',
  components: {
    TrinityRingsSpinner
  },
  data() {
    return {
      file: null,
      show_result: false,
      loading: false,
      preview: '',
      imageSrc: '',
    }
  },
  methods: {
    handleFileChange(event) {
      this.file = event.target.files[0];
      this.preview = URL.createObjectURL(this.file);
      console.log(this.file);
    },
    analyzeFile() {
      this.loading = true;
      const formData = new FormData();
        formData.append('file', this.file);
        const headers = { 
          'Content-Type': 'multipart/form-data',
        };
        axios.post('http://localhost:5000/analyze', formData, { headers, responseType: 'arraybuffer' }).then((res) => {
          const uint8Array = new Uint8Array(res.data);
          const binaryData = uint8Array.reduce((acc, byte) => acc + String.fromCharCode(byte), '');
          const imageBase64 = btoa(binaryData);
          this.imageSrc = 'data:image/png;base64,' + imageBase64;
          this.show_result = true;
          this.loading = false;
      });
    },
    getImageBlob (image_url) {
      return axios.get(image_url).then(response => window.URL.createObjectURL(response.data))
    },
    resetAnalyze() {
      this.show_result = false;
    }
  }
}
</script>