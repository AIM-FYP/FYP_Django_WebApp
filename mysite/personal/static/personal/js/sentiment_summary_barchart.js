
    /* Group By time - count(pos),count(neg),count(neu) */
    /* Displays number of sentiment counts for a candidate selected  */
 
console.log('bingo!');

x=['x'];

positives=[];
positives.push('pos');

negatives=[];
negatives.push('neg')

neutrals=[];
neutrals.push('neu')

console.log("b4func",sentiment_summary_linechart_Data);

for (i in sentiment_summary_linechart_Data){
    console.log("Tweet: ",sentiment_summary_linechart_Data[i]);
    positives.push(sentiment_summary_linechart_Data[i].pos);
    negatives.push(sentiment_summary_linechart_Data[i].neg);
    neutrals.push(sentiment_summary_linechart_Data[i].neu);
    x.push(sentiment_summary_linechart_Data[i].time);
}
console.log(negatives);
console.log(positives);
console.log(neutrals);
console.log(x);
/* Sentiment Summary Barchart */


var chart = c3.generate({
    bindto: '#chart1',
    data: {
        x: 'x',
        xFormat: '%H', // 'xFormat' can be used as custom format of 'x'
        columns: [
            x,
            positives,
            negatives,
            neutrals
        ]
    },
    axis: {
        x: {
            type: 'timeseries',
            tick: {
                count:function(d){
                    if ((sentiment_summary_linechart_Data.length)<=5){
                        return sentiment_summary_linechart_Data.length;
                    }
                    else{
                        return 5;
                    } 
                },
                format: '%H:00'
            }
        }
    },
    size: {
        width: 300,
        height: 300
    },
    padding: {
       
      right: 20
    },
    color: {
        pattern: ['#38ed62', '#e35e57', '#4a69e6']
    }
});

