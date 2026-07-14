const imageInput=document.getElementById("imageInput");

const preview=document.getElementById("preview");

if(imageInput){

imageInput.onchange=function(){

const file=this.files[0];

if(file){

preview.src=URL.createObjectURL(file);

preview.style.display="block";

}

}

}