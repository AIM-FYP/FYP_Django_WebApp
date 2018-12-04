/*  Sentiment Summary Donut!  */
var Data=[
  {  /* from collected percent positive percent negative percent neutral */
      "pos":"50",
      "neg":"30",
      "neu":"20"
  }
];










var chart = c3.generate({
bindto: '#donut',
data: {
    columns: [
        ['Positive', 50],
        ['Negative', 30],
        ['Neutral', 20]
    ],
    type : 'donut'//,
    //onclick: function (d, i) { console.log("onclick", d, i); },
    //onmouseover: function (d, i) { console.log("onmouseover", d, i); },
    //onmouseout: function (d, i) { console.log("onmouseout", d, i); }
},
legend:{
    item:{onclick : function(d){}}
},
donut: {
    //title: "Text in mid circle of donut"
    width: 30,
    label: {
        show: false
      }
},
color: {
    pattern: ['#38ed62', '#e35e57', '#4a69e6']

},
size: {
    height: 250
}
});


/*some chart unloaing
setTimeout(function () {
chart.unload({
    ids: 'data1'
});
chart.unload({
    ids: 'data2'
});
}, 50);
*/

