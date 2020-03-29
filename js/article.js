// 保护知识产权
if(document.domain=='localhost'){
    var copyright="";
    // 如果在首页无法找到copyright的话
    if(window.location.protocol + '//' + window.location.host+'/'!=window.document.location.href){
        copyright=$(".copyright")[0]['innerText'];
    } 
    else{
        copyright="文章作者:崔恩博\r\n\r\n作者链接:https://cccceb.github.io/\r\n\r\n许可协议: '署名-非商用-相同方式共享 4.0' 转载请保留原文链接及作者。"
    }
    $(document).bind({  
        copy: function(e) {//copy事件 
            var cpTxt =copyright; 
            var clipboardData = window.clipboardData; //for IE  
            if (!clipboardData) { // for chrome  
                clipboardData = e.originalEvent.clipboardData;  
            }
            console.log("copy success.");
        
            //替换
            var selection = window.getSelection().toString();
            if(selection.length<15){
                clipboardData.setData('Text',selection); 
                return false;
            }
            if(e.clipboardData){
                e.clipboardData.setData('text/plain', selection + copyright);
            }else if(window.clipboardData){
                //ie浏览器
                window.clipboardData.setData('text/plain', selection + copyright);
            } 
            clipboardData.setData('Text',selection+'\r\n\r\n'+copyright);  
            return false;//否则设不生效  
        }
    });
}

