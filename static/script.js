var video=[]
var imageScaleFactor = 0.5;
var outputStride = 16;
var flipHorizontal = false;

var id;


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
         p.setAttribute("height",640)
         p.setAttribute("width",480)
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




document.getElementById("btn").addEventListener("click",is_human);


var i=0;
var a;
var img_arr=[];
var fl=0;
var x;

function is_human(){
  var canvas = document.getElementById("canvas0")
  var ratio=ratio = video[0].videoWidth/video[0].videoHeight;
  var w = video[0].videoWidth-100;
  var h = parseInt(w/ratio,10);
  canvas.width = w;
  canvas.height = h;

canvas.getContext('2d').drawImage(video[0], 0, 0,w,h);
var imageElement = canvas;
var img=canvas.toDataURL();
posenet.load().then(function(net){
      return net.estimateSinglePose(imageElement, imageScaleFactor, flipHorizontal, outputStride)
    }).then(function(pose){
    	console.log(pose)
      if(pose.score>0.5){
      	if(!a)
      	a=setInterval(gait,100);
      fl=img_arr.length;
      console.log("final length : "+fl);
      }
      else{
      	if(a){
      			clearInterval(a);
            console.log(img_arr)
            url="http://127.0.0.1:5000/gait";
            var data={
              x:img_arr.slice(0,fl)
            }
            $.post(url,data,function(v,status){
              id=v;
              face()
              console.log(v);
            })
           

      			return;
      	}

      }
      is_human();
    })

}
    

 function gait(){

  var canvas = document.getElementById("canvas0")
  var ratio=ratio = video[0].videoWidth/video[0].videoHeight;
  var w = video[0].videoWidth-100;
  var h = parseInt(w/ratio,10);
  canvas.width = w;
  canvas.height = h;

canvas.getContext('2d').drawImage(video[0], 0, 0,w,h);
var imageElement = canvas;
var img=canvas.toDataURL();

img_arr.push(img);
console.log(img_arr.length);
 }







var x=10;

 function face(){
  var canvas1 = document.getElementById("canvas1")
  var canvas2 = document.getElementById("canvas2")
  var ratio= video[1].videoWidth/video[1].videoHeight;
  var w = video[1].videoWidth-100;
  var h = parseInt(w/ratio,10);
  canvas1.width = w;
  canvas1.height = h;
  canvas2.width = w;
  canvas2.height = h;

canvas1.getContext('2d').drawImage(video[1], 0, 0,w,h);
var img1=canvas1.toDataURL();

canvas2.getContext('2d').drawImage(video[2], 0, 0,w,h);
var img2=canvas2.toDataURL();



  url="feedFace";
  if (x > 0) {
    var data={
      'x1':img1,
      'x2':img2
    }
    $.post(url,data,function(v,status){
      console.log(v);
    })
    x--;
  } else {
    url = 'getFace';
    $.get(url, function(v, status) {
      console.log(v);
    })
  }

}