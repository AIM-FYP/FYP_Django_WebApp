/*  Sentiment Summary Donut! */ 


/* from collected percent positive percent negative percent neutral */

var chart = c3.generate({
bindto: '#donut',
data: {
    columns: [
        ['Positive', sentiment_summary_donut_Data[0]['pos']],
        ['Negative', sentiment_summary_donut_Data[0]['neg']],
        ['Neutral' , sentiment_summary_donut_Data[0]['neu']]
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
    height: 300,
    width : 300
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

