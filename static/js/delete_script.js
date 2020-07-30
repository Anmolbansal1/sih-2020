var names = [];

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

function remove(i) {
   console.log(names[i - 1]);
   var data = {
      "user": names[i - 1]
   }
   $('#loader').show();
   $.post("delete", data, function (v, status) {
      console.log(v);
      $('#loader').hide();
      fillTable();
   })
}

function fillTable() {
   $.get("delete", function (res) {
      let t_row = template `
         }
         <tr>
         <th scope="row">${'id'}</th>
         <td>${'name'}</td>
         <td>${'button'}</td>
         </tr>`;
      let i = 1;
      names = res['data'];
      $("#table-result-rows").empty();
      res.data.forEach(function (name) {
         var txt = t_row({
            id: i,
            name: name,
            button: "<button class=\"vid btn btn-danger\" onclick=\"remove(" + i + ")\">" + "Delete" + "</button>"
         })
         $("#table-result-rows").append(txt);
         i++;
      });
   })
}

fillTable();