/*  Sentiment Summary Guage!  */


var Data=[{
    /* link is used to get average sentiment link is https://cp.hana.ondemand.com/dps/d/preview/6e5cad3a964d429194c436c18a184f6f/1605%20500/en-US/frameset.htm?104cf9550e5d7e43e10000000a4450e5.html  */
    'Sentiment':"50"   
}];








var chart = c3.generate({
    bindto: '#guage',
    data: {
        columns: [
            ['Sentiment',50]
        ],
        type: 'gauge'//,
        //onclick: function (d, i) { console.log("onclick", d, i); },
        //onmouseover: function (d, i) { console.log("onmouseover", d, i); },
        //onmouseout: function (d, i) { console.log("onmouseout", d, i); }
    },
    legend: {
        show: false
    },
    gauge: {
        label: {
            format: function(value, ratio) {
                return value+'%';
            },
            show: true // to turn off the min/max labels.
        },
    min: 0, // 0 is default, //can handle negative min e.g. vacuum / voltage / current flow / rate of change
    max: 100, // 100 is default

    width: 30 // for adjusting arc thickness
    },
    color: {
        pattern: ['#e35e57', '#4a69e6', '#38ed62'], // the three color levels for the percentage values.
        threshold: {
//            unit: 'value', // percentage is default
//            max: 200, // 100 is default
            values: [40, 70, 100]
        }
    },
    size: {
        height: 180
    }
});
//Update function
/*
setTimeout(function () {
    chart.load({
        columns: [['Sentiment', 10]]
    });
}, 1000);
*/
