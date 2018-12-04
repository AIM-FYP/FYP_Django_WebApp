/* Raw Tweets script */
Data = [
    {
    "sentiment":"pos",
    "percent":"50",
    "text":"Hi, I am a pos tweet! wecwed wjednwedjwed wejdnjwkendew djewndjkenwd ewjdnejkwndjkew dewkdjnwjkendkjenwd ejkwdnkjwe", 
    "time":"19:03"
    },
    {
    "sentiment":"pos",
    "percent":"50",
    "text":"Hi, I am a pos tweet!", 
    "time":"19:03"    
    },
    {
    "sentiment":"pos",
    "percent":"50",
    "text":"Hi, I am a pos tweet!" , 
    "time":"19:03"   
    },
    {
    "sentiment":"pos",
    "percent":"50",
    "text":"Hi, I am a pos tweet!" , 
    "time":"19:03"   
    },
    {
    "sentiment":"neu",
    "percent":"50",
    "text":"Hi, I am a neu tweet!" , 
    "time":"19:03"   
    },
    {
    "sentiment":"neg",
    "percent":"50",
    "text":"Hi, I am a neg tweet!" , 
    "time":"19:03"   
    },
    {
    "sentiment":"neu",
    "percent":"50",
    "text":"Hi, I am a neu tweet!" , 
    "time":"19:03"   
    },
    {
    "sentiment":"pos",
    "percent":"50",
    "text":"Hi, I am a pos tweet!", 
    "time":"19:03"    
    },
    {
    "sentiment":"neg",
    "percent":"50",
    "text":"Hi, I am a neg tweet!", 
    "time":"19:03"    
    },
    {
    "sentiment":"pos",
    "percent":"50",
    "text":"Hi, I am a pos tweet!", 
    "time":"19:03"    
    },
    {
    "sentiment":"neg",
    "percent":"50",
    "text":"Hi, I am a neg tweet!", 
    "time":"19:03"    
    }

];



d3.select("body")
.select("#rawtweets")
.select("div.row")
.select("#leftdiv")
.select("div.divTable,div.blueTable")
.select("div.divTableBody")
.selectAll("div.divTableRow")
.data(Data).enter()
.append("div")
    .attr("class","divTableRow")
    .attr("id","demo")
    .append("div")
    .attr("class","divTableCell")
    .html(function(d){return "<h8 style='font-weight:bold;float: right;font-family: serif; font-size: 13px;color: green'>"+d.percent+"%"+"</h8>"
    +"<br>"+d.text
    +"<br>"+"<h8 style='margin-top: 4px;float:right;font-size: 9px;color:grey;float:right;'>"
    +"<img src='/static/personal/img/clock.png' alt='Clock IMG' style='margin-right: 2px;float:left;width:14px;height:14px;'>"+d.time+"</h8>"
    +"<br>"});


d3.select("body")
.select("#rawtweets")
.select("div.row")
.select("#middiv")
.select("div.divTable,div.blueTable")
.select("div.divTableBody")
.selectAll("div.divTableRow")
.data(Data).enter()
.append("div")
    .attr("class","divTableRow")
    .attr("id","demo")
    .append("div")
    .attr("class","divTableCell")
    .html(function(d){return "<h8 style='font-weight:bold;float: right;font-family: serif; font-size: 13px;color: green'>"+d.percent+"%"+"</h8>"
    +"<br>"+d.text
    +"<br>"+"<h8 style='margin-top: 4px;float:right;font-size: 9px;color:grey;float:right;'>"
    +"<img src='/static/personal/img/clock.png' alt='Clock IMG' style='margin-right: 2px;float:left;width:14px;height:14px;'>"+d.time+"</h8>"
    +"<br>"});


d3.select("body")
.select("#rawtweets")
.select("div.row")
.select("#rightdiv")
.select("div.divTable,div.blueTable")
.select("div.divTableBody")
.selectAll("div.divTableRow")
.data(Data).enter()
.append("div")
    .attr("class","divTableRow")
    .attr("id","demo")
    .append("div")
    .attr("class","divTableCell")
    .html(function(d){return "<h8 style='font-weight:bold;float: right;font-family: serif; font-size: 13px;color: green'>"+d.percent+"%"+"</h8>"
    +"<br>"+d.text
    +"<br>"+"<h8 style='margin-top: 4px;float:right;font-size: 9px;color:grey;float:right;'>"
    +"<img src='/static/personal/img/clock.png' %}' alt='Clock IMG' style='margin-right: 2px;float:left;width:14px;height:14px;'>"+d.time+"</h8>"
    +"<br>"});

