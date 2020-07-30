var namelist = []


var searchbox = document.getElementById("autocomplete")

var searchlist = []

searchbox.addEventListener('keypress', function (e) {
	if (e.key === 'Enter') {
		if (namelist.indexOf(searchbox.value) == -1)
			alert("no such name exist")
		else if (searchlist.indexOf(searchbox.value) == -1)
			add_list(searchbox.value)
		else
			alert("name already in search list")

		console.log(searchlist)
		searchbox.value = ""
	}
});

document.getElementById("modal-videos-close").addEventListener("click", closeModal)

function closeModal() {
	// $("#modal-videos-holder").empty();
	$('#myModal').modal('toggle');
}


$.get("userlist", function (v, status) {

	namelist = v.users

	$("#autocomplete").autocomplete({
		source: namelist
	});

})

function remove_me(e) {
	console.log($(e)[0].innerText)

	searchlist.splice(searchlist.indexOf($(e)[0].innerText), 1)

	console.log(searchlist)

	$(e).remove()

}

function add_list(name) {
	searchlist.push(name);
	document.getElementById("feed_info").setAttribute("style", "display: block;")
	document.getElementById("selectedPersonsHelpBlock").setAttribute("style", "display: block;")

	var txt = "<button class=\"btn btn-info mr-2\" onclick=\"remove_me(this)\">" + name + "</button>";
	console.log(searchlist);
	console.log(txt);
	$("#feed_list").append(txt)
}

document.getElementById("submit").addEventListener("click", search)
// imported
function template(strings, ...keys) {
	return (function (...values) {
		let dict = values[values.length - 1] || {};
		let result = [strings[0]];
		keys.forEach(function (key, i) {
			let value = Number.isInteger(key) ? values[key] : dict[key];
			result.push(value, strings[i + 1]);
		});
		return result.join('');
	});
}

function search() {
	var st = document.getElementById("start_time").value
	var et = document.getElementById("end_time").value
	var data = {
		"start": st,
		"end": et,
		"list": searchlist
	}

	let t_row = template `
	<tr>
		<th scope="row">${'id'}</th>
		<td>${'name'}</td>
     	<td>${'time'}</td>
     	<td>${'button'}</td>
     </tr>`;

	$.post("finder", data, function (v, status) {
		dirs = v;
		console.log(v.file)
		document.getElementById("table-results").setAttribute("style", "display: contents;");
		$("#table-result-rows").empty();
		for (var i = 0; i < v.file.length; i++) {
			var txt = t_row({
				'id': (i + 1),
				name: v.file[i][0],
				time: v.file[i][3],
				button: "<button class=\"vid btn btn-info\" onclick=\"getVid(" + i + ")\">" + "get videos" + "</button>"
			})
			console.log(txt);
			$("#table-result-rows").append(txt)
		}
	})
}

var dirs = []

function getVid(index) {
	$('#myModal').modal('toggle');
	var data = {
		"url": [dirs.file[index][1], dirs.file[index][2]]
	}

	$.post("video_feed", data, function (v, status) {

		console.log(v)
		document.getElementById("vid1").src = v.urls[0];
		document.getElementById("vid2").src = v.urls[1];
		// var txt = "<video src=\"" + v.urls[0] + "\" controls></video>";
		// $("#modal-videos-holder").append(txt)
		// var txt = "<video src=\"" + v.urls[1] + "\" controls></video>";
		// $("#modal-videos-holder").append(txt)
	})
}
