import { Input } from '@nextui-org/react';
import { useFormContext } from 'react-hook-form';

import { WaterProperties } from './WaterForm';

interface FieldProps {
  label: string;
  property: any;
}

export const Field = ({ label, property }: FieldProps) => {
  const {
    register,
    formState: {
      errors: { Turbidity: error },
      isSubmitting,
    },
  } = useFormContext<WaterProperties>();

  return (
    <Input
      bordered
      clearable
      helperColor="error"
      color="primary"
      status={error ? 'error' : 'default'}
      fullWidth
      size="lg"
      disabled={isSubmitting}
      labelPlaceholder={label}
      {...register(property, {
        required: {
          value: true,
          message: `${label} required`,
        },
      })}
    />
  );
};
