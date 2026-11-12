(function () {
  if (!window.echarts) return;
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  var fontFamily = '"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",system-ui,sans-serif';

  /* ---------- 图 2：优化前 namespace 分布 ---------- */
  var nsEl = document.getElementById('chart-ns');
  if (nsEl) {
    var nsChart = echarts.init(nsEl, null, { renderer: 'svg' });
    var nsData = [
      { name: '旧 · default_topic（十六进制编码）', value: 2118 },
      { name: '旧 · 默认主题（中文编码）', value: 785 },
      { name: '旧 · 随机副本 A', value: 231 },
      { name: '旧 · 随机副本 B', value: 91 },
      { name: '旧 · 随机副本 C', value: 91 },
      { name: '旧 · 随机副本 D', value: 91 },
      { name: '旧 · 随机副本 E', value: 91 },
      { name: '目标 · default_topic（稳定房间）', value: 23 },
      { name: '旧 · 随机副本 F', value: 17 }
    ];
    nsChart.setOption({
      animation: false,
      backgroundColor: 'transparent',
      textStyle: { fontFamily: fontFamily, color: ink },
      grid: { left: 210, right: 70, top: 24, bottom: 30 },
      tooltip: {
        trigger: 'axis',
        appendToBody: true,
        axisPointer: { type: 'shadow' },
        formatter: function (p) {
          var d = p[0];
          return d.name + '<br/>记录数：<strong>' + d.value + '</strong> 条';
        }
      },
      xAxis: {
        type: 'value',
        name: '向量记录数（条）',
        nameLocation: 'end',
        nameGap: 18,
        nameTextStyle: { color: muted, fontSize: 11 },
        axisLine: { lineStyle: { color: rule } },
        axisLabel: { color: muted, fontSize: 11 },
        splitLine: { lineStyle: { color: rule, opacity: 0.6 } }
      },
      yAxis: {
        type: 'category',
        inverse: true,
        data: nsData.map(function (d) { return d.name; }),
        axisLine: { lineStyle: { color: rule } },
        axisTick: { show: false },
        axisLabel: { color: ink, fontSize: 11 }
      },
      series: [{
        type: 'bar',
        barWidth: '58%',
        data: nsData.map(function (d, i) {
          var isTarget = d.name.indexOf('目标') === 0;
          return {
            value: d.value,
            itemStyle: {
              color: isTarget ? accent2 : accent,
              opacity: isTarget ? 1 : 0.82,
              borderRadius: [0, 4, 4, 0]
            }
          };
        }),
        label: {
          show: true,
          position: 'right',
          color: ink,
          fontSize: 11,
          fontWeight: 700,
          formatter: function (p) { return p.value; }
        }
      }]
    });
    window.addEventListener('resize', function () { nsChart.resize(); });
  }

  /* ---------- 图 3：优化前后能力具备情况 ---------- */
  var capEl = document.getElementById('chart-cap');
  if (capEl) {
    var capChart = echarts.init(capEl, null, { renderer: 'svg' });
    var capNames = [
      '多篇对比覆盖',
      '刷新后数据可见',
      '跨工作区隔离',
      '扫描版 PDF 可读',
      '失败可诊断',
      '重资源缓存'
    ];
    var before = capNames.map(function () { return 0; });
    var after = capNames.map(function () { return 1; });

    capChart.setOption({
      animation: false,
      backgroundColor: 'transparent',
      textStyle: { fontFamily: fontFamily, color: ink },
      legend: {
        top: 4,
        left: 210,
        itemWidth: 12,
        itemHeight: 12,
        textStyle: { color: muted, fontSize: 11 }
      },
      grid: { left: 210, right: 60, top: 44, bottom: 34 },
      tooltip: {
        trigger: 'axis',
        appendToBody: true,
        axisPointer: { type: 'shadow' },
        formatter: function (p) {
          var lines = [p[0].name];
          p.forEach(function (it) {
            lines.push(it.seriesName + '：' + (it.value === 1 ? '具备' : '不具备'));
          });
          return lines.join('<br/>');
        }
      },
      xAxis: {
        type: 'value',
        min: 0,
        max: 1.2,
        interval: 1,
        axisLine: { lineStyle: { color: rule } },
        axisLabel: {
          color: muted,
          fontSize: 11,
          formatter: function (v) { return v === 0 ? '不具备' : (v === 1 ? '具备' : ''); }
        },
        splitLine: { lineStyle: { color: rule, opacity: 0.6 } }
      },
      yAxis: {
        type: 'category',
        inverse: true,
        data: capNames,
        axisLine: { lineStyle: { color: rule } },
        axisTick: { show: false },
        axisLabel: { color: ink, fontSize: 12 }
      },
      series: [
        {
          name: '优化前',
          type: 'bar',
          barWidth: '38%',
          data: before,
          itemStyle: { color: muted, opacity: 0.55, borderRadius: [0, 4, 4, 0] },
          label: {
            show: true,
            position: 'right',
            color: muted,
            fontSize: 11,
            formatter: function () { return '0'; }
          }
        },
        {
          name: '优化后',
          type: 'bar',
          barWidth: '38%',
          data: after,
          itemStyle: { color: accent2, borderRadius: [0, 4, 4, 0] },
          label: {
            show: true,
            position: 'right',
            color: ink,
            fontSize: 11,
            fontWeight: 700,
            formatter: function () { return '1'; }
          }
        }
      ]
    });
    window.addEventListener('resize', function () { capChart.resize(); });
  }
})();
