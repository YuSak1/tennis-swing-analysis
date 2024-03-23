// Loading animation
var buttonNode = document.querySelector('#loading-button');
buttonNode.addEventListener('click', function(e){
    showLoading();
}, false);

function showLoading(){
    var loadingNode = document.querySelector('#loading');
    var loadingNode_msg = document.querySelector('#loading_msg');
    var flash = document.querySelector('#flash');
    loadingNode.style.display = '';
    loadingNode_msg.style.display = '';
    flash.style.display = 'none';
}