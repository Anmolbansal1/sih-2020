var video=[]
const loadlModelPromise = cocoSsd.load();
var THRESHOLD_HUMAN=0.5
var id;
var gaits = [document.getElementById("gait1")]

var done=0;

var con = document.getElementById("hide");

var i=0;

navigator.mediaDevices.enumerateDevices()
.then(function(devices) {
  devices.forEach(function(device) {
   if(device.kind=="videoinput"){
    console.log(device.kind + ": " + device.label +
                " id = " + device.deviceId);
    var constraints = { deviceId: { exact: device.deviceId } };
    navigator.getUserMedia({ audio: false, video: constraints },
      function(stream) {
         var para = document.createElement("video");
         var p = document.createElement("canvas");
         para.setAttribute("id","video"+i)
         p.setAttribute("id","canvas"+i)
         p.setAttribute("height",480)
         p.setAttribute("width",640)
         p.setAttribute("style", "display: none;")
         para.setAttribute("height",480)
         para.setAttribute("width",640)
         con.appendChild(para)
         con.appendChild(p)
         document.getElementById("video"+i).srcObject = stream;
         document.getElementById("video"+i).play()
         video.push(document.getElementById("video"+i))
         i++;
      },
      function(err) {
         console.log("The following error occurred: " + err.name);
      }
   );
   }
  });
})


var img_arr=[];
var bb_arr=[];
var ff = 0;





document.getElementById("btn").addEventListener("click",run_gait);


//gait functions


function run_gait(){

  console.log("starting gait")

      // define a Promise that'll be used to load the model
     
      
      // resolve all the Promises
      Promise.all([loadlModelPromise])
        .then(values => {
          console.log("gait mode loaded")
          console.log(values)
          detectFromVideoFrame(values[0], document.getElementById("gait1"));
        })
        .catch(error => {
          console.error(error);
        });
    
  }

 detectFromVideoFrame = (model, image) => {
  console.log("detecting" )
    model.detect(image).then(predictions => {

      const ctx = document.getElementById("canvas1");
      ctx.getContext("2d").drawImage(image,0, 0, ctx.width, ctx.height);
      //console.log(predictions)
      is_human(ctx.toDataURL(),predictions);

      dispImg(model,image)

    }, (error) => {
      console.log("Couldn't start the webcam")
      console.error(error)
    });


  };

function is_human(img,pred){
  x=find(pred)
  if(x==null||x.score<THRESHOLD_HUMAN){
    if(img_arr.length==0) return;
    ff--;
    if(ff==0){
      send_gait();
    }
  }
  else{
    console.log(img_arr.length)
    img_arr.push(img);
    bb_arr=bb_arr.concat(x.bbox);
    ff=5;
  }
}

function find(pred){
  var x;
  for(var i=0;i<pred.length;i++){
    if(pred[i].class=="person"){
      if(!x||pred[i].score>x.score)
        x=pred[i];
    }
  }
  return x;
}

function send_gait(){
  var data={
    "x" : img_arr,
    "bb": bb_arr
  }

console.log(data)


if(img_arr.length<10){
  console.log("false alarm gait");

img_arr=[];
bb_arr=[];
  return;
}

feed_face()

$.post("gait",data,function(v,status){
    console.log("***************gait results ***************")
      console.log(v);
       if(done==1){
      done=0;
     $.get("final",function(v,status){
        grant_access(v.final_answer)
        console.log(v);
      })
     return;
   }
   done=1;

      
    })

img_arr=[];
bb_arr=[];

}



function dispImg(model,image){
  let data;
fetch("http://192.168.1.7:8080/photo.jpg").then(response => {
  response.blob().then(blobResponse => {
    data = blobResponse;
    reader.readAsDataURL(data); 
    const urlCreator = window.URL || window.webkitURL;
      if(model!=null)
      detectFromVideoFrame(model, image)
      else 
        dispImg()
  })
});
}

dispImg();

var reader = new FileReader();
 reader.onloadend = function() {
     var base64data = reader.result;   
    gaits[0].src=base64data          
 }


 

//face functions


var counter=10;

function feed_face(){
console.log("feeding faces")
const ctx = document.getElementById("canvas0");
ctx.getContext("2d").drawImage(video[0],0, 0, ctx.width, ctx.height);
var img = ctx.toDataURL()

var data={
  "x":img
}

$.post("feedFace",data,function(v,status){
      console.log(v);
      counter--;
      if(counter!=0)
        feed_face();
      else 
        final_face()
    })

}

function final_face(){
  counter=10;
  console.log("final face")
  $.get("getFace",function(v,status){
    if(v.flag==0){
      feed_face()
      return;
    }
    console.log(v)
    if(done==1){
      done=0;
     $.get("final",function(v,status){
        grant_access(v.final_answer)
        console.log(v);
      })
     return;
   }
   done=1;
     
    })

}


function grant_access(name){
  if(name=="unknown"){
    document.getElementById("user").innerHTML="Unknown person entering ";
    do_something()
  }
   else{
    display(name)
    data={
      "user":name
    }
    $.post("access",data,function(v,status){
      console.log(v)
    })
   }
}

function do_something(){
  document.getElementById('allow').disabled=false;
  document.getElementById('deny').disabled=false;
}

document.getElementById("allow").addEventListener("click",get_user);
document.getElementById("deny").addEventListener("click",deny_user);
document.getElementById("go").addEventListener("click",allow_user);

function get_user(){

  // add diplay
  document.getElementById('user_name').setAttribute("style", "display: block;")
  document.getElementById('go').setAttribute("style", "display: block;")
  document.getElementById('user_name').disabled=false;
  document.getElementById('go').disabled=false;

}

function deny_user(){
  grant_access("terrorist")
  document.getElementById("user").innerHTML=""
  document.getElementById('allow').disabled=true;
  document.getElementById('go').disabled=true;
  document.getElementById('deny').disabled=true;
  document.getElementById('user_name').disabled=true;
  document.getElementById('user_name').value="";
}

function allow_user(){
  x=document.getElementById('user_name').value;
  grant_access(x)
  document.getElementById('allow').disabled=true;
  document.getElementById('go').disabled=true;
  document.getElementById('deny').disabled=true;
  document.getElementById('user_name').disabled=true;
  document.getElementById('user_name').value="";
}


function display(name){
  console.log("diplaying")
  if(name!="terrorist")
document.getElementById("user").innerHTML="Access Granted to "+name;

setTimeout(function(){ document.getElementById("user").innerHTML="" }, 20000);
}


document.getElementById("train_it").addEventListener("click",call_train);

function call_train(){
  
  $.get("train",function(v,status){
    console.log(v)
  })
}
