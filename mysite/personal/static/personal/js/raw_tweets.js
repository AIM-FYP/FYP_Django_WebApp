/* Raw Tweets script 
Data = [
    {
    "sentiment":"pos",
    "percent":"50",
    "text":"Hi, I am a tweet!", 
    "time":"19:03"
    },
    {
    "sentiment":"pos",
    "percent":"50",
    "text":"Hi, I am a tweet!", 
    "time":"19:03"    
    },
    {
    "sentiment":"pos",
    "percent":"50",
    "text":"Hi, I am a tweet!" , 
    "time":"19:03"   
    },
    {
    "sentiment":"pos",
    "percent":"50",
    "text":"Hi, I am a tweet!" , 
    "time":"19:03"   
    },
    {
    "sentiment":"neu",
    "percent":"50",
    "text":"Hi, I am a tweet!" , 
    "time":"19:03"   
    }

];
*/
console.log('155px');
_positives=[];
_negatives=[];
_neutrals=[];
for (i in raw_tweets_Data){
    if (raw_tweets_Data[i].sentiment=="pos"){
        _positives.push(raw_tweets_Data[i]);
    }
    if (raw_tweets_Data[i].sentiment=="neg"){
        _negatives.push(raw_tweets_Data[i]);
    }
    if (raw_tweets_Data[i].sentiment=="neu"){
        _neutrals.push(raw_tweets_Data[i]);
    }
}

d3.select("body")
.select("#rawtweets")
.select("div.row")
.select("#leftdiv")
.select("div.divTable,div.blueTable")
.select("div.divTableBody")
.selectAll("div.divTableRow")
.data(_positives).enter()
.append("div")
    .attr("class","divTableRow")
    .attr("id","demo")
    .append("div")
    .attr("class","divTableCell")
    //.attr("style", "overflow:hidden;")
    .attr("style","height:200px;position:relative;")
  
    .html(function(d){return "<h8 style='font-weight:bold;float: right;font-family: serif; font-size: 13px;color: green'>"+d.percentage+"%"+"</h8>"
    +"<br>"+"<h8 style='margin: 0;position: absolute;top: 50%;left: 25%;transform: translate(-20%, -50%);'>"+d.text+"</h8>"
    +"<h8 style='margin-top: 155px;float:right;font-size: 9px;color:grey;float:right;'>"
    +"<img src='/static/personal/img/clock.png' alt='Clock IMG' style='margin-right: 2px;float:left;width:14px;height:14px;'>"+d.time+"</h8>"
    });
                   


d3.select("body")
.select("#rawtweets")
.select("div.row")
.select("#middiv")
.select("div.divTable,div.blueTable")
.select("div.divTableBody")
.selectAll("div.divTableRow")
.data(_negatives).enter()
.append("div")
    .attr("class","divTableRow")
    .attr("id","demo")
    .append("div")
    .attr("class","divTableCell")
    //.attr("style", "overflow:hidden;")
    .attr("style","height:200px;position:relative;")

    .html(function(d){return "<h8 style='font-weight:bold;float: right;font-family: serif; font-size: 13px;color: #e35e57'>"+d.percentage+"%"+"</h8>"
    +"<br>"+"<h8 style=' margin: 0;position: absolute;top: 50%;left: 25%;transform: translate(-20%, -50%);'>"+d.text+"</h8>"
    +"<h8 style='margin-top: 155px;float:right;font-size: 9px;color:grey;float:right;'>"
    +"<img src='/static/personal/img/clock.png' alt='Clock IMG' style='margin-right: 2px;float:left;width:14px;height:14px;'>"+d.time+"</h8>"
    });
                   


d3.select("body")
.select("#rawtweets")
.select("div.row")
.select("#rightdiv")
.select("div.divTable,div.blueTable")
.select("div.divTableBody")
.selectAll("div.divTableRow")
.data(_neutrals).enter()
.append("div")
    .attr("class","divTableRow")
    .attr("id","demo")
    .append("div")
    .attr("class","divTableCell")
    //.attr("style", "overflow:hidden;")
    .attr("style","height:200px;position:relative;")

    .html(function(d){return "<h8 style='font-weight:bold;float: right;font-family: serif; font-size: 13px;color: #4a69e6'>"+d.percentage+"%"+"</h8>"
    +"<br>"+"<h8 style='margin: 0;position: absolute;top: 50%;left: 25%;transform: translate(-20%, -50%);'>"+d.text+"</h8>"
    +"<h8 style='margin-top: 155px;float:right;font-size: 9px;color:grey;float:right;'>"
    +"<img src='/static/personal/img/clock.png' %}' alt='Clock IMG' style='margin-right: 2px;float:left;width:14px;height:14px;'>"+d.time+"</h8>"
    });
               


/*


d3.select("body")
.append("div")
.attr("style","height:200px;position:relative;border:3px solid green;")
.html("<h8 style='margin:0;position:absolute;top:50%;left:50%;transform:translate(-50%, -50%);'> okokokokokokokok </h8>")
*/