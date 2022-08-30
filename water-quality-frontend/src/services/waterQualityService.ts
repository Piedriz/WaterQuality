import Swal from 'sweetalert2';

import { WaterProperties } from '../components/WaterForm';

export const getWaterQuality = async (waterProperties: WaterProperties) => {
  const apiUrl = 'http://localhost:8888';
  const response = await fetch(
    `${apiUrl}/water?ph=${waterProperties.ph}&Hardness=${waterProperties.Hardness}&Solids=${waterProperties.Solids}&Chloramines=${waterProperties.Chloramines}&Sulfate=${waterProperties.Sulfate}&Conductivity=${waterProperties.Conductivity}&Organic_carbon=${waterProperties.Organic_carbon}&Trihalomethanes=${waterProperties.Trihalomethanes}&Turbidity=${waterProperties.Turbidity}`,
  );
  const quality = +(await response.text())[1];

  let text = 'Potable';
  let icon: any = 'success';
  if (quality == 0) {
    text = 'No potable';
    icon = 'error';
  }
  Swal.fire({ title: 'Resultado', text, icon });
};
