(this.webpackJsonpreact=this.webpackJsonpreact||[]).push([[0],{66:function(e,t,a){e.exports=a(93)},71:function(e,t,a){},93:function(e,t,a){"use strict";a.r(t);var n=a(0),l=a.n(n),r=a(7),c=a.n(r),o=a(11),i=(a(71),a(131)),u=a(133),s=a(52),p=a.n(s),m=a(57),d=a(125),E=a(132),b=a(134),f=function(){var e=Object(m.a)(),t=e.errors,a=e.register,r=e.handleSubmit,c=Object(n.useState)(""),s=Object(o.a)(c,2),f=s[0],h=s[1],g=Object(n.useState)(!1),v=Object(o.a)(g,2),y=v[0],j=v[1],I=Object(n.useState)("predictPost"),O=Object(o.a)(I,2),w=O[0],P=O[1];return l.a.createElement("div",{className:"App"},l.a.createElement("h1",null,"NLP App"),l.a.createElement("p",null,"Enter issue description and app will tell you which group shall resolve it"),l.a.createElement("form",{onSubmit:r((function(e){console.log(e),h(""),j(!0),console.log("dropdown selected",w),p.a.post("/"+w,{query:e.inputfield}).then((function(e){console.log("response",e),e.data&&h(e.data.group),j(!1)}),(function(e){console.log("error",e),j(!1)}))}))},l.a.createElement(E.a,{name:"algo",ref:a,labelId:"demo-customized-select-label",id:"demo-customized-select",value:w,onChange:function(e){P(e.target.value)}},l.a.createElement(b.a,{value:"predictPost"},l.a.createElement("em",null,"FastText")),l.a.createElement(b.a,{value:"predictBert"},"Bert"),l.a.createElement(b.a,{value:"predictDirt"},"Dirt"),l.a.createElement(b.a,{value:"PredictGotIt"},"GotIt")),l.a.createElement(u.a,{inputRef:a({required:!0}),name:"inputfield",id:"standard-full-width",label:"Issue Description",style:{margin:8},placeholder:"Enter Description here",fullWidth:!0,margin:"normal",InputLabelProps:{shrink:!0}}),t.inputfield&&"Input cannot be empty",l.a.createElement("br",null),l.a.createElement(i.a,{color:"primary",type:"submit"},"Get Group"),l.a.createElement("br",null),y?l.a.createElement(d.a,{color:"secondary"}):null),l.a.createElement("p",null,f))},h=document.getElementById("root");c.a.render(l.a.createElement(f,null),h)}},[[66,1,2]]]);
//# sourceMappingURL=main.2c6049b0.chunk.js.map