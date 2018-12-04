var Data = [
    /* Group By time - count(pos),count(neg),count(neu) */
    /* Displays number of sentiment counts for a candidate selected  */
    { 
        "pos":"32",
        "neg":"76",
        "neu":"54",
        "time":"19:02"
    },
    { 
        "pos":"32",
        "neg":"76",
        "neu":"54",
        "time":"19:03"
    },
    { 
        "pos":"32",
        "neg":"76",
        "neu":"54",
        "time":"19:04"
    },
    { 
        "pos":"32",
        "neg":"76",
        "neu":"54",
        "time":"19:05"
    },
    { 
        "pos":"32",
        "neg":"76",
        "neu":"54",
        "time":"19:06"
    },
    { 
        "pos":"32",
        "neg":"76",
        "neu":"54",
        "time":"19:07"
    }    
];






/* Sentiment Summary Barchart */
var chart = c3.generate({
    bindto: '#chart1',
    data: {
        columns: [
            ['pos', 67,56,43,45,67,87,65,45,67,43,23,56,87],
            ['neg', 89,34,54,34,65,34,55,67,65,88,77,55,43],
            ['neu', 12,21,23,12,32,45,23,43,32,12,32,43,54]
        ],
        types: {
            pos: 'spline',
            neg: 'spline',
            neu: 'spline',
        }
    },
    size: {
        width: 300,
        height: 300
    },
    padding: {
        /*20 b4*/
      right: 20
    },
    color: {
        pattern: ['#38ed62', '#e35e57', '#4a69e6']

    }

});


